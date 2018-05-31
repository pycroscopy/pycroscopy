def gsf_read(file_name):
    '''Read a Gwyddion Simple Field 1.0 file format
    http://gwyddion.net/documentation/user-guide-en/gsf.html
    
    Args:
        file_name (string): the name of the output (any extension will be replaced)
    Returns:
        metadata (dict): additional metadata to be included in the file
        data (2darray): an arbitrary sized 2D array of arbitrary numeric type
    '''
    if file_name.rpartition('.')[1] == '.':
        file_name = file_name[0:file_name.rfind('.')]
    
    gsfFile = open(file_name + '.gsf', 'rb')
    
    metadata = {}
    
    # check if header is OK
    if not(gsfFile.readline().decode('UTF-8') == 'Gwyddion Simple Field 1.0\n'):
        gsfFile.close()
        raise ValueError('File has wrong header')
        
    term = b'00'
    # read metadata header
    while term != b'\x00':
        line_string = gsfFile.readline().decode('UTF-8')
        metadata[line_string.rpartition(' = ')[0]] = line_string.rpartition('=')[2]
        term = gsfFile.read(1)
        gsfFile.seek(-1, 1)
    
    gsfFile.read(4 - gsfFile.tell() % 4)
    
    #fix known metadata types from .gsf file specs
    #first the mandatory ones...
    metadata['XRes'] = np.int(metadata['XRes'])
    metadata['YRes'] = np.int(metadata['YRes'])
    
    #now check for the optional ones
    if 'XReal' in metadata:
        metadata['XReal'] = np.float(metadata['XReal'])
    
    if 'YReal' in metadata:
        metadata['YReal'] = np.float(metadata['YReal'])
                
    if 'XOffset' in metadata:
        metadata['XOffset'] = np.float(metadata['XOffset'])
    
    if 'YOffset' in metadata:
        metadata['YOffset'] = np.float(metadata['YOffset'])
    
    data = np.frombuffer(gsfFile.read(),dtype='float32').reshape(metadata['YRes'],metadata['XRes'])
    
    gsfFile.close()
    
    return metadata, data
