"""
Created on Mar 14, 2016

@author: Chris Smith -- csmith55@utk.edu
"""
import os
import datetime
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='PySPM Versioner', description='Increments the version number for PySPM')
    parser.add_argument('--manual',type=str,default=None)
    parser.add_argument('--minor',default=False,action='store_true')
    parser.add_argument('--major',default=False,action='store_true')
    args = parser.parse_args()
    new_minor = args.minor
    new_major = args.major
    
    main_dir = os.getcwd()
    self_name = 'versioner.py'
    new_main_version = False
    for root, dirs, files in os.walk(main_dir):
        rootname = os.path.split(root)[-1]
        if rootname == 'scripts':
            continue
        if 'external_libs' in root and rootname != 'external_libs':
            continue
        new_version = False
        vc_time = 0
        vm_time = 0
        v_time = 0
        dir_version = [0,0,0]
#         print root, 'contains'
#         print 'directories', dirs
#         print 'files', files
        f_time = 0
        for file in files:
            if file.split('.')[1] == 'pyc':
                continue
            elif file == self_name:
                continue
            file_path = os.path.join(root,file)
            f_stats = os.stat(file_path)
            if file == '__version__.py':
                vc_time = f_stats[9]
                vm_time = f_stats[8]
                v_time = max([vc_time, vm_time])
            fc_time = f_stats[9]
            fm_time = f_stats[8]
            f_time = max([fc_time,fm_time,f_time])
#             print '{} was created {} and modified {}.'.format(file, fc_time, fm_time)
            
        if v_time == 0:
#             print 'No version file found.  Will create new at version 0.0.1'
            v_file = open(os.path.join(root,'__version__.py'),'w')
            new_version = True
        elif root == main_dir:
            continue
        else:
#             print 'Version file found.  Reading old version number.'
            v_file = open(os.path.join(root,'__version__.py'),'r+')
            for iline, line in enumerate(v_file.readlines()):
                line = line.split()
                if line == []:
                    continue
                elif line[0] == 'major':
                    dir_version[0] = int(line[-1])
                elif line[0] == 'minor':
                    dir_version[1] = int(line[-1])
                elif line[0] == 'micro':
                    dir_version[2] = int(line[-1])                                                            
                else:
                    continue
                
            new_version = f_time < v_time
        
        if dir_version == [0,0,0]:
#             print 'Blank version file.'
            new_version = True
#         print 'Old version was {}.{}.{}'.format(*dir_version)
        if new_version:
            print 'Files have been modified since last version.'
            new_main_version = True
            dir_version[2]+=1
            if new_minor:
                dir_version[1]+= 1
                dir_version[2] = 0
            if new_major:
                dir_version[0]+= 1
                dir_version[1] = 0
                dir_version[2] = 0
            v_file.seek(0)
            v_file.write('major\t=\t{}\n'.format(dir_version[0]))
            v_file.write('minor\t=\t{}\n'.format(dir_version[1]))
            v_file.write('micro\t=\t{}\n'.format(dir_version[2]))
            print 'New version is {}.{}.{}'.format(*dir_version)
        
        v_file.close()
        
    if new_main_version:
        mv_file = open(os.path.join(main_dir,'__version__.py'),'r+')
        mv_version = '0.0.1'
        ver_date = datetime.datetime.now()
        mv_date = '{}/{}/{}'.format(ver_date.month,ver_date.day,ver_date.year)
        for line in mv_file.readlines():
            print line
            if line.strip() == '':
                continue
            field, value = line.split('=')
            if field == 'version':
                mv_version = value.strip['\'']
            elif field == 'date':
                mv_date = value.strip['\'']
            elif field == 'time':
                mv_time = value.strip['\'']
        
        mv_major,mv_minor,mv_micro = mv_version.split('.')
        mv_micro = str(int(mv_micro)+1)
        
        if new_minor:
            mv_minor+= 1
            mv_micro = 0
        if new_major:
            mv_major+= 1
            mv_minor = 0
            mv_micro = 0
        
        if args.manual:
            mv_version = args.manual
        else:
            mv_version = '.'.join([mv_major,mv_minor,mv_micro])
        mv_time = '{}:{}:{}'.format(ver_date.hour,ver_date.minute,ver_date.second)
        mv_file.seek(0)
        mv_file.truncate()
        mv_file.write('version = {} \n'.format(mv_version))
        mv_file.write('date = {} \n'.format(mv_date))
        mv_file.write('time = {} \n'.format(mv_time))
        
        print 'New main version is {}'.format(mv_version)
        print 'Vesion date is {}'.format(mv_date)
                        
    pass                