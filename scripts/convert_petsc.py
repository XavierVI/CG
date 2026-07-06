import scipy.io, PetscBinaryIO
import os
import glob

def convert(file_in, file_out):
    A = scipy.io.mmread(file_in) # this supports .mtx files
    A = A.tocsr()
    PetscBinaryIO.PetscBinaryIO().writeMatSciPy(open(file_out,'w'), A)


root_path = os.path.join(os.getcwd(), '..', 'matrices')
mtx_root_path = os.environ['HOME'] + '/workspace/aCG/mtx_files'
files = glob.glob(mtx_root_path + '/*/*.mtx')

print('Converting all .mtx files in ' + mtx_root_path + ' to .petsc format')
print('Files found: ' + str(len(files)))
print(f'File type: {type(files[0])}')

os.makedirs(root_path, exist_ok=True)

for file in files:
    if file.endswith('.mtx'):
        file_in = os.path.join(mtx_root_path, file)
        file_out = os.path.basename(file_in.replace('.mtx', '.petsc'))
        file_out = os.path.join(root_path, file_out)
        print('Converting ' + file_in + ' to ' + file_out)
        convert(file_in, file_out)