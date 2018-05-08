"""
Created on Aug 7, 2015

@author: James Anderson

link: https://github.com/jamesra/dm4reader
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import collections
import struct
import array

DM4Header = collections.namedtuple('DM4Header', ('version', 'root_length', 'little_endian'))
DM4TagHeader = collections.namedtuple('DM4TagHeader', ('type', 'name', 'byte_length', 'array_length',
                                                       'data_type_code', 'header_offset', 'data_offset'))
DM4DirHeader = collections.namedtuple('DM4DirHeader', ('type', 'name', 'byte_length', 'sorted', 'closed',
                                                       'num_tags', 'data_offset'))

DM4Tag = collections.namedtuple('DM4Tag', ('name', 'data_type_code', 'data'))

DM4DataType = collections.namedtuple('DM4DataTypes', ('num_bytes', 'signed', 'type_format'))

DM4DataTypeDict = {2: DM4DataType(2, True, 'h'),  # 2byte signed integer
                   3: DM4DataType(4, True, 'i'),  # 4byte signed integer
                   4: DM4DataType(2, False, 'H'),  # 2byte unsigned integer
                   5: DM4DataType(4, False, 'I'),  # 4byte unsigned integer
                   6: DM4DataType(4, False, 'f'),  # 4byte float
                   7: DM4DataType(8, False, 'd'),  # 8byte float
                   8: DM4DataType(1, False, '?'),
                   9: DM4DataType(1, False, 'c'),
                   10: DM4DataType(1, True, 'b'),
                   11: DM4DataType(8, True, 'q'),
                   12: DM4DataType(8, True, 'Q')
                   }

DM4_header_size = 4 + 8 + 4
DM4_root_tag_dir_header_size = 1 + 1 + 8


def tag_is_directory(tag):
    return tag.type == 20


def read_header_dm4(dmfile):
    dmfile.seek(0)
    version = struct.unpack_from('>I', dmfile.read(4))[0]  # int.from_bytes(dmfile.read(4), byteorder='big')
    rootlength = struct.unpack_from('>Q', dmfile.read(8))[0]
    byteorder = struct.unpack_from('>I', dmfile.read(4))[0]

    little_endian = byteorder == 1

    return DM4Header(version, rootlength, little_endian)


def _get_endian_str(endian):
    """
    DM4 header encodes little endian as byte value 1 in the header
    :return: 'big' or 'little' for use with python's int.frombytes function
    """
    if isinstance(endian, str):
        return endian

    assert (isinstance(endian, int))
    if endian == 1:
        return 'little'

    return 'big'


def _get_struct_endian_str(endian):
    """
    DM4 header encodes little endian as byte value 1 in the header.  However when that convention is followed the wrong
    values are read.  So this implementation is reversed.
    :return: '>' or '<' for use with python's struct.unpack function
    """
    if isinstance(endian, str):
        if endian == 'little':
            return '>'  # Little Endian
        else:
            return '<'  # Big Endian
    else:
        if endian == 1:
            return '>'  # Little Endian
        else:
            return '<'  # Big Endian


def read_root_tag_dir_header_dm4(dmfile, endian):
    """Read the root directory information from a dm4 file.
       File seek position is left at end of root_tag_dir_header"""
    if not isinstance(endian, str):
        endian = _get_struct_endian_str(endian)

    dmfile.seek(DM4_header_size)

    issorted = struct.unpack_from(endian + 'b', dmfile.read(1))[0]
    isclosed = struct.unpack_from(endian + 'b', dmfile.read(1))[0]
    num_tags = struct.unpack_from('>Q', dmfile.read(8))[0]  # DM4 specifies this property as always big endian

    return DM4DirHeader(20, None, 0, issorted, isclosed, num_tags, DM4_header_size)


def read_tag_header_dm4(dmfile, endian):
    """Read the tag from the file.  Leaves file at the end of the tag data, ready to read the next tag from the file"""
    tag_header_offset = dmfile.tell()
    tag_type = struct.unpack_from(endian + 'B', dmfile.read(1))[0]
    if tag_type == 20:
        return _read_tag_dir_header_dm4(dmfile, endian)
    if tag_type == 0:
        return None

    tag_name = _read_tag_name(dmfile, endian)
    tag_byte_length = struct.unpack_from('>Q', dmfile.read(8))[0]  # DM4 specifies this property as always big endian

    tag_data_offset = dmfile.tell()

    _read_tag_garbage_str(dmfile)

    (tag_array_length, tag_array_types) = _read_tag_data_info(dmfile)

    dmfile.seek(tag_data_offset + tag_byte_length)
    return DM4TagHeader(tag_type, tag_name, tag_byte_length, tag_array_length, tag_array_types[0], tag_header_offset,
                        tag_data_offset)


def _read_tag_name(dmfile, endian):
    tag_name_len = struct.unpack_from('>H', dmfile.read(2))[0]  # DM4 specifies this property as always big endian
    tag_name = None
    if tag_name_len > 0:
        data = dmfile.read(tag_name_len)
        try:
            tag_name = data.decode('utf-8', errors='ignore')
        except UnicodeDecodeError as e:
            tag_name = None
            pass

    return tag_name


def _read_tag_dir_header_dm4(dmfile, endian):
    tag_name = _read_tag_name(dmfile, endian)
    tag_byte_length = struct.unpack_from('>Q', dmfile.read(8))[0]  # DM4 specifies this property as always big endian
    issorted = struct.unpack_from(endian + 'b', dmfile.read(1))[0]
    isclosed = struct.unpack_from(endian + 'b', dmfile.read(1))[0]
    num_tags = struct.unpack_from('>Q', dmfile.read(8))[0]  # DM4 specifies this property as always big endian

    data_offset = dmfile.tell()

    return DM4DirHeader(20, tag_name, tag_byte_length, issorted, isclosed, num_tags, data_offset)


def _read_tag_garbage_str(dmfile):
    """
    DM4 has four bytes of % symbols in the tag.  Ensure it is there.
    """
    garbage_str = dmfile.read(4).decode('utf-8')
    assert (garbage_str == '%%%%')


def _read_tag_data_info(dmfile):
    tag_array_length = struct.unpack_from('>Q', dmfile.read(8))[0]  # DM4 specifies this property as always big endian
    format_str = '>' + tag_array_length * 'q'  # Big endian signed long

    tag_array_types = struct.unpack_from(format_str, dmfile.read(8 * tag_array_length))

    return tag_array_length, tag_array_types


def _read_tag_data(dmfile, tag, endian):
    assert (tag.type == 21)
    try:

        endian = _get_struct_endian_str(endian)
        dmfile.seek(tag.data_offset)

        _read_tag_garbage_str(dmfile)
        (tag_array_length, tag_array_types) = _read_tag_data_info(dmfile)

        tag_data_type_code = tag_array_types[0]

        if tag_data_type_code == 15:
            return read_tag_data_group(dmfile, tag, endian)
        elif tag_data_type_code == 20:
            return read_tag_data_array(dmfile, tag, endian)

        if tag_data_type_code not in DM4DataTypeDict:
            print("Missing type " + str(tag_data_type_code))
            return None

        return _read_tag_data_value(dmfile, endian, tag_data_type_code)

    finally:
        # Ensure we are in the correct position to read the next tag regardless of how reading this tag goes
        dmfile.seek(tag.data_offset + tag.byte_length)


def _read_tag_data_value(dmfile, endian, type_code):
    data_type = DM4DataTypeDict[type_code]
    format_str = _get_struct_endian_str(endian) + data_type.type_format
    byte_data = dmfile.read(data_type.num_bytes)

    return struct.unpack_from(format_str, byte_data)[0]


def read_tag_data_group(dmfile, tag, endian):
    endian = _get_struct_endian_str(endian)
    dmfile.seek(tag.data_offset)

    _read_tag_garbage_str(dmfile)
    (tag_array_length, tag_array_types) = _read_tag_data_info(dmfile)

    tag_data_type = tag_array_types[0]
    assert (tag_data_type == 15)

    length_groupname = tag_array_types[1]
    number_of_entries_in_group = tag_array_types[2]
    field_data = tag_array_types[3:]

    field_types_list = []

    for iField in range(0, number_of_entries_in_group):
        fieldname_length = field_data[iField * 2]
        fieldname_type = field_data[(iField * 2) + 1]
        field_types_list.append(fieldname_type)

    fields_data = []
    for field_type in field_types_list:
        field_data = _read_tag_data_value(dmfile, endian, field_type)
        fields_data.append(field_data)

    return fields_data


def read_tag_data_array(dmfile, tag, endian):
    endian = _get_struct_endian_str(endian)
    dmfile.seek(tag.data_offset)

    _read_tag_garbage_str(dmfile)

    (tag_array_length, tag_array_types) = _read_tag_data_info(dmfile)

    assert (tag_array_types[0] == 20)
    array_data_type_code = tag_array_types[1]
    array_length = tag_array_types[2]

    if array_data_type_code == 15:
        return "Array of groups length %d and type %d" % (array_length, array_data_type_code)

    assert (len(tag_array_types) == 3)

    data_type = DM4DataTypeDict[array_data_type_code]

    data = array.array(data_type.type_format)
    data.fromfile(dmfile, array_length)
    return data


class DM4File:
    @property
    def endian_str(self):
        return self._endian_str

    @property
    def hfile(self):
        return self._hfile

    def __init__(self, filedata):
        """
        :param file filedata: file handle to dm4 file
        """
        self._hfile = filedata
        self.header = read_header_dm4(self.hfile)
        self._endian_str = _get_struct_endian_str(self.header.little_endian)

        self.root_tag_dir_header = read_root_tag_dir_header_dm4(self.hfile, endian=self.endian_str)

    def close(self):
        self._hfile.close()
        self._hfile = None

    @classmethod
    def open(cls, filename):
        """
        :param str filename: Name of DM4 file to open
        :rtype: DM4File
        :return: DM4File object
        """

        hfile = open(filename, "rb")
        return DM4File(hfile)

    def read_tag_data(self, tag):
        """Read the data associated with the passed tag"""
        return _read_tag_data(self.hfile, tag, self.endian_str)

    DM4TagDir = collections.namedtuple('DM4Dir',
                                       ('name', 'dm4_tag', 'named_subdirs',
                                        'unnamed_subdirs', 'named_tags', 'unnamed_tags'))

    def read_directory(self, directory_tag=None):
        """
        Read the directories and tags from a dm4 file.  The first step in working with a dm4 file.
        :return: A named collection containing information about the directory
        """

        if directory_tag is None:
            directory_tag = self.root_tag_dir_header

        dir_obj = DM4File.DM4TagDir(directory_tag.name, directory_tag, {}, [], {}, [])

        for iTag in range(0, directory_tag.num_tags):
            tag = read_tag_header_dm4(self.hfile, self.endian_str)
            if tag is None:
                break

            if tag_is_directory(tag):
                if tag.name is None:
                    dir_obj.unnamed_subdirs.append(self.read_directory(tag))
                else:
                    dir_obj.named_subdirs[tag.name] = self.read_directory(tag)
            else:
                if tag.name is None:
                    dir_obj.unnamed_tags.append(tag)
                else:
                    dir_obj.named_tags[tag.name] = tag

        return dir_obj
