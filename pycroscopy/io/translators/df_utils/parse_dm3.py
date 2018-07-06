# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 07:55:56 2016

@author: Chris Smith
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import struct
import array
import warnings
import re

try:
    import StringIO.StringIO
except ImportError:
    from io import StringIO

    unicode = str

# mfm 2013-05-21 do we need the numpy array stuff? The python array module
# allows us to store arrays easily and efficiently. How do we deal
# with arrays of complex data? We could use numpy arrays with custom dtypes
# in which case we'd be totally tied to numpy, or else stick with structarray.
# Either way the current setup of treating arrays like numpy arrays in a few
# special cases isn't particularly nice, that could be done outside this module
# and then we'd only have to check for the basic types.
# NB our struct array class is not that different from a very basic array.array
# it has data and a list of data types. We could just store bytes in the data
# instead of lists of tuples?

# mfm 9Feb13 Simpler version than v1, but maybe less robust
# (v1 cnows about names we're trying to extract at extraction time
# this one doesn't). Is easier to follow though
verbose = False
# if we find data which matches this regex we return a
# string instead of an array
treat_as_string_names = ['.*Name']


def get_from_file(f, stype):
    # print("reading", stype, "size", struct.calcsize(stype))
    src = f.read(struct.calcsize(stype))
    assert (len(src) == struct.calcsize(stype))
    d = struct.unpack(stype, src)
    if verbose:
        print(d)
    d = [d1.decode('utf-8', 'ignore') if isinstance(d1, bytes) else d1 for d1 in d]
    if len(d) == 1:
        return d[0]
    else:
        return d


def put_into_file(f, stype, *args):
    f.write(struct.pack(
        stype, *args))


read_array = lambda f, a, l: a.fromstring(f.read(l * struct.calcsize(a.typecode))) \
    if isinstance(f, StringIO) else a.fromfile(f, l)
write_array = lambda f, a: f.write(a.tostring()) if isinstance(f, StringIO) else a.tofile(f)


class structarray(object):
    """
    A class to represent struct arrays. We store the data as a list of
    tuples, with the dm_types telling us the dm id for the  types
    """

    def __init__(self, typecodes):
        # self.dm_types = dm_types
        self.typecodes = typecodes
        self.raw_data = None

    def __eq__(self, other):
        return self.raw_data == other.raw_data and self.typecodes == other.typecodes

    def __repr__(self):
        return "structarray({}, {})".format(self.typecodes, self.raw_data)

    def bytelen(self, num_elements):
        return num_elements * struct.calcsize(" ".join(self.typecodes))

    def num_elements(self):
        b = self.bytelen(1)
        assert (len(self.raw_data) % b == 0)
        return len(self.raw_data) // b

    def from_file(self, f, num_elements):
        self.raw_data = f.read(self.bytelen(num_elements))

    def to_file(self, f):
        f.write(self.raw_data)


def parse_dm_header(f, outdata=None):
    """
    This is the start of the DM file. We check for some
    magic values and then treat the next entry as a tag_root

    If outdata is supplied, we write instead of read using the dictionary outdata as a source
    Hopefully parse_dm_header(newf, outdata=parse_dm_header(f)) copies f to newf
    """
    # filesize is sizeondisk - 16. But we have 8 bytes of zero at the end of
    # the file.
    if outdata is not None:
        version, file_size, endianness = 3, -1, 1
        put_into_file(f, "> l l l", version, file_size, endianness)
        start = f.tell()
        parse_dm_tag_root(f, outdata)
        end = f.tell()
        # start is end of 3 long header. We want to write 2nd long
        f.seek(start - 8)
        # the real file size. We started counting after 12-byte version,fs,end
        # and we need to subtract 16 total:
        put_into_file(f, "> l", end - start + 4)
        f.seek(end)
        enda, endb = 0, 0
        put_into_file(f, "> l l", enda, endb)
    else:
        version, file_size, endianness = get_from_file(f, "> l l l")
        assert (version == 3)
        assert (endianness == 1)
        start = f.tell()
        ret = parse_dm_tag_root(f, outdata)
        end = f.tell()
        # print("fs", file_size, end - start, (end-start)%8)
        # mfm 2013-07-11 the file_size value is not always
        # end-start, sometimes there seems to be an extra 4 bytes,
        # other times not. Let's just ignore it for the moment
        # assert(file_size == end - start)
        enda, endb = get_from_file(f, "> l l")
        assert (enda == endb == 0)
        return ret


def parse_dm_tag_root(f, outdata=None):
    if outdata is not None:
        is_dict = 0 if isinstance(outdata, list) else 1
        _open, num_tags = 0, len(outdata)
        put_into_file(f, "> b b l", is_dict, _open, num_tags)
        if not is_dict:
            if verbose:
                print("list:", outdata)
            for subdata in outdata:
                parse_dm_tag_entry(f, subdata, None)
        else:
            if verbose:
                print("dict:", outdata)
            for key in outdata:
                if verbose:
                    print("Writing", key, outdata[key])
                assert (key is not None)
                parse_dm_tag_entry(f, outdata[key], key)
    else:
        is_dict, _open, num_tags = get_from_file(f, "> b b l")
        if verbose:
            print("New tag root", is_dict, _open, num_tags)
        if is_dict:
            new_obj = {}
            for i in range(num_tags):
                name, data = parse_dm_tag_entry(f)
                assert (name is not None)
                if verbose:
                    print("Read name", name, "at", f.tell())
                new_obj[name] = data
        else:
            new_obj = []
            for i in range(num_tags):
                name, data = parse_dm_tag_entry(f)
                assert (name is None)
                if verbose:
                    print("appending...", i, "at", f.tell())
                new_obj.append(data)

        return new_obj


def parse_dm_tag_entry(f, outdata=None, outname=None):
    if outdata is not None:
        dtype = 20 if isinstance(outdata, (dict, list)) else 21
        name_len = len(outname) if outname else 0
        put_into_file(f, "> b H", dtype, name_len)
        if outname:
            put_into_file(f, ">" + str(name_len) + "s", outname)

        if dtype == 21:
            parse_dm_tag_data(f, outdata)
        else:
            parse_dm_tag_root(f, outdata)

    else:
        dtype, name_len = get_from_file(f, "> b H")
        if name_len:
            name = get_from_file(f, ">" + str(name_len) + "s")
        else:
            name = None
        if dtype == 21:
            arr = parse_dm_tag_data(f)
            if name and hasattr(arr, "__len__") and len(arr) > 0:
                for regex in treat_as_string_names:
                    if re.match(regex, name):
                        if isinstance(arr[0], int):
                            arr = ''.join(chr(x) for x in arr)
                        elif isinstance(arr[0], str):
                            arr = ''.join(arr)

            return name, arr
        elif dtype == 20:
            return name, parse_dm_tag_root(f)
        else:
            raise Exception("Unknown data type=" + str(dtype))


def parse_dm_tag_data(f, outdata=None):
    # todo what is id??
    # it is normally one of 1,3,7,11,19
    # we can parse lists of numbers with them all 1
    # strings work with 3
    # could id be some offset to the start of the data?
    # for simple types we just read data, for strings, we read type, length
    # for structs we read len,num, len0,type0,len1,... =num*2+2
    # structs (15) can be 7,9,11,19
    # arrays (20) can be 3 or 11
    if outdata is not None:
        # can we get away with a limited set that we write?
        # ie can all numbers be doubles or ints, and we have lists
        _, data_type = get_structdmtypes_for_python_typeorobject(outdata)
        if verbose:
            print("treating {} as {}".format(outdata, data_type))
        if not data_type:
            raise Exception("Unsupported type: {}".format(type(outdata)))
        _delim = "%%%%"
        put_into_file(f, "> 4s l l", _delim, 0, data_type)
        pos = f.tell()
        header = dm_types[data_type](f, outdata)
        f.seek(pos - 8)  # where our header_len starts
        put_into_file(f, "> l", header + 1)
        f.seek(0, 2)
    else:
        _delim, header_len, data_type = get_from_file(f, "> 4s l l")
        assert (_delim == "%%%%")
        ret, header = dm_types[data_type](f)
        assert (header + 1 == header_len)
        return ret


# we store the id as a key and the name,
# struct format, python types in a tuple for the value
# mfm 2013-08-02 was using l, L for long and ulong but sizes vary
# on platforms
# can we use i, I instead?
dm_simple_names = {
    2: ("short", "h", []),
    3: ("long", "i", [int]),
    4: ("ushort", "H", []),
    5: ("ulong", "I", [int]),
    6: ("float", "f", []),
    7: ("double", "d", [float]),
    8: ("bool", "b", [bool]),
    9: ("char", "b", []),
    10: ("octet", "b", [])
}

dm_complex_names = {
    18: "string",
    15: "struct",
    20: "array"}


def get_dmtype_for_name(name):
    for key, (_name, sc, types) in dm_simple_names.items():
        if _name == name:
            return key
    for key, _name in dm_complex_names.items():
        if _name == name:
            return key
    return 0


def get_structdmtypes_for_python_typeorobject(typeorobj):
    """
    Return structchar, dmtype for the python (or numpy)
    type or object typeorobj.
    For more complex types we only return the dm type
    """
    # not isinstance is probably a bit more lenient than 'is'
    # ie isinstance(x,str) is nicer than type(x) is str.
    # hence we use isinstance when available
    if isinstance(typeorobj, type):
        comparer = lambda test: test is typeorobj
    else:
        comparer = lambda test: isinstance(typeorobj, test)

    for key, (name, sc, types) in dm_simple_names.items():
        for t in types:
            if comparer(t):
                return sc, key
    if comparer(str):
        return None, get_dmtype_for_name('array')  # treat all strings as arrays!
    elif comparer(array.array):
        return None, get_dmtype_for_name('array')
    elif comparer(tuple):
        return None, get_dmtype_for_name('struct')
    elif comparer(structarray):
        return None, get_dmtype_for_name('array')
    warnings.warn("No appropriate DMType found for %s, %s", typeorobj, type(typeorobj))
    return None


def get_structchar_for_dmtype(dm_type):
    name, sc, types = dm_simple_names[dm_type]
    return sc


def get_dmtype_for_structchar(struct_char):
    for key, (name, sc, types) in dm_simple_names.items():
        if struct_char == sc:
            return key
    return -1


def standard_dm_read(datatype_num, desc):
    """
    datatype_num is the number of the data type, see dm_simple_names
    above. desc is a (nicename, struct_char) tuple. We return a function
    that parses the data for us.
    """
    nicename, structchar, types = desc

    def dm_read_x(f, outdata=None):
        """Reads (or write id outdata is given) a simple data type.
        returns the data if reading and the number of bytes of header
        """
        if outdata is not None:
            put_into_file(f, "<" + structchar, outdata)
            return 0
        else:
            return get_from_file(f, "<" + structchar), 0

    return dm_read_x


dm_types = {}
for key, val in dm_simple_names.items():
    dm_types[key] = standard_dm_read(key, val)


# 8 is boolean, and relatively easy:


def dm_read_bool(f, outdata=None):
    if outdata:
        put_into_file(f, "<b", 1 if outdata else 0)
        return 0
    else:
        return get_from_file(f, "<b") != 0, 0


dm_types[get_dmtype_for_name('bool')] = dm_read_bool


# string is 18:


# mfm 2013-05-13 looks like this is never used, and all strings are
# treated as array?
def dm_read_string(f, outdata=None):
    header_size = 1  # just a length field
    if outdata is not None:
        outdata = outdata.encode("utf_16_le")
        slen = len(outdata)
        put_into_file(f, ">L", slen)
        put_into_file(f, ">" + str(slen) + "s", outdata)
        return header_size
    else:
        assert False
        slen = get_from_file(f, ">L")
        raws = get_from_file(f, ">" + str(slen) + "s")
        if verbose:
            print("Got String", unicode(raws, "utf_16_le"), "at", f.tell())
        return unicode(raws, "utf_16_le"), header_size


dm_types[get_dmtype_for_name('string')] = dm_read_string


# struct is 15
def dm_read_struct_types(f, outtypes=None):
    if outtypes is not None:
        _len, nfields = 0, len(outtypes)
        put_into_file(f, "> l l", _len, nfields)
        for t in outtypes:
            _len = 0
            put_into_file(f, "> l l", _len, t)
        return 2 + 2 * len(outtypes)
    else:
        types = []
        _len, nfields = get_from_file(f, "> l l")
        assert (_len == 0)  # is it always?
        for i in range(nfields):
            _len, dtype = get_from_file(f, "> l l")
            types.append(dtype)
            assert (_len == 0)
            assert (dtype != 15)  # we don't allow structs of structs?
        return types, 2 + 2 * nfields


def dm_read_struct(f, outdata=None):
    if outdata is not None:
        start = f.tell()
        types = [get_structdmtypes_for_python_typeorobject(x)[1]
                 for x in outdata]
        header = dm_read_struct_types(f, types)
        for t, data in zip(types, outdata):
            dm_types[t](f, data)
        # we write length at the very end
        # but _len is probably not len, it's set to 0 for the
        # file I'm trying...
        write_len = False
        if write_len:
            end = f.tell()
            f.seek(start)
            # dm_read_struct first writes a length which we overwrite here
            # I think the length ignores the length field (4 bytes)
            put_into_file(f, "> l", end - start - 4)
            f.seek(0, 2)  # the very end (2 is pos from end)
            assert (f.tell() == end)
        return header
    else:
        types, header = dm_read_struct_types(f)
        if verbose:
            print("Found struct with types", types, "at", f.tell())

        ret = []
        for t in types:
            d, h = dm_types[t](f)
            ret.append(d)
        return tuple(ret), header


dm_types[get_dmtype_for_name('struct')] = dm_read_struct


# array is 20
def dm_read_array(f, outdata=None):
    array_header = 2  # type, length
    if outdata is not None:
        if isinstance(outdata, structarray):
            # we write type, struct_types, length
            put_into_file(f, "> l", get_dmtype_for_name('struct'))
            outdmtypes = [get_dmtype_for_structchar(s) for s in outdata.typecodes]
            struct_header = dm_read_struct_types(f, outtypes=outdmtypes)
            put_into_file(f, "> L", outdata.num_elements())
            outdata.to_file(f)
            return struct_header + array_header
        elif isinstance(outdata, (str, unicode, array.array)):
            if isinstance(outdata, (str, unicode)):
                outdata = array.array('H', outdata.encode("utf_16_le"))
            assert (isinstance(outdata, array.array))
            dtype = get_dmtype_for_structchar(outdata.typecode)
            put_into_file(f, "> l", dtype)
            put_into_file(f, "> L", len(outdata))
            write_array(f, outdata)
            return array_header
        else:
            warnings.warn("Unsupported type for conversion to array:%s", outdata)

    else:
        # supports arrays of structs and arrays of types,
        # but not arrays of arrays (Is this possible)
        # actually lets just use the array object, which only allows arrays of
        # simple types!

        # arrays of structs are pretty common, eg in a simple image CLUT
        # data["DocumentObjectList"][0]["ImageDisplayInfo"]["CLUT"] is an
        # array of 3 bytes
        # we can't handle arrays of structs easily, as we use lists for
        # taglists, dicts for taggroups and arrays for array data.
        # But array.array only supports simple types. We need a new type, then.
        # let's make a structarray
        dtype = get_from_file(f, "> l")
        if dtype == get_dmtype_for_name('struct'):
            types, struct_header = dm_read_struct_types(f)
            alen = get_from_file(f, "> L")
            if verbose:
                print(types)
                print("Array of structs! types %s, len %d" % (",".join(map(str, types)), alen), "at", f.tell())
            ret = structarray([get_structchar_for_dmtype(d) for d in types])
            ret.from_file(f, alen)
            return ret, array_header + struct_header
        else:
            # mfm 2013-08-02 struct.calcsize('l') is 4 on win and 8 on Mac!
            # however >l, <l is 4 on both... could be a bug?
            # Can we get around this by adding '>' to out structchar?
            # nope, array only takes a sinlge char. Trying i, I instead
            struct_char = get_structchar_for_dmtype(dtype)
            ret = array.array(struct_char)
            alen = get_from_file(f, "> L")
            if verbose:
                print("Array type %d len %d struct %c size %d" % (
                    dtype, alen, struct_char, struct.calcsize(struct_char)), "at", f.tell())
            if alen:
                # faster to read <1024f than <f 1024 times. probly
                # stype = "<" + str(alen) + dm_simple_names[dtype][1]
                # ret = get_from_file(f, stype)
                read_array(f, ret, alen)
            if verbose:
                print("Done Array type %d len %d" % (dtype, alen), "at", f.tell())
            return ret, array_header


dm_types[get_dmtype_for_name('array')] = dm_read_array
