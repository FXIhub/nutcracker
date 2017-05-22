import numpy as np
from mulpro import mulpro
import os
import h5py
import nutcracker.utils.rotate

class ErrorMatrixBruteForce:
    """
    Class for a multiprocessed brute force algorith, which gives the errror matrix as an output.

    Args:
        :model1_filename(str):                location of a hdf5 file which contains the model
        :model2_filename(str):                location of a hdf5 file which contains the model
        :model1_dataset(str):                 dataset of the file
        :model2_dataset(str):                 dataset of the file 

    Kwargs:
        :number_of_processes(int):            number of processes for multiprocessing, default=1
        :chunck_size(int):                    size of the search chunck, default=10
        :number_of_evaluations(int):          number of grid points on which the brute force optimises, default=10
        :order_spline_interpolation(int):     the order of the spline interpolation, has to be in range 0-5, default = 3 [from scipy.org]
        :cropping_model(int):                 cropps the model by the given vaule in total, has to be an even number, default = 0
        :mask(bool ndarray):                  provide a mask to be used for the evaluation of the cost function, default = None
        :radius_radial_mask(int):             applies a radial mask to the model with given radius, default = None
        :search_range(float/list):            absolute angle in radian in which the optimisation should be done, default = np.pi/2.
    """

    # initialise the class
    def __init__(self,
                 model1_filename,
                 model2_filename,
                 model1_dataset,
                 model2_dataset,
                 number_of_processes=1,
                 chunck_size=10,
                 number_of_evaluations=10,
                 order_spline_interpolation=3,
                 cropping_model=None,
                 mask=None,
                 radius_radial_mask=None,
                 search_range=np.pi/2.):

        # global variables to define number of tasks and counter
        self.number_of_processes = number_of_processes
        self.counter = 0
        self.n_tasks = number_of_evaluations**3 / chunck_size

        # loads the models if possible
        if model1_filename and model2_filename and model1_dataset and model2_dataset:
            self.get_models(model1_filename,model2_filename,model1_dataset,model2_dataset)
        else:
            print 'no proper filename or dataset given'


    def get_models(self,model1_filename,model2_filename,model1_dataset,model2_dataset):
        # loading the models
        with h5py.File(model1_filename, 'r') as f:
            self.model1 = f[model1_dataset][:]
        with h5py.File(model2_filename, 'r') as f:
            self.model2 = f[model2_dataset][:]
        return self.model1, self.model2

    def get_chunck(self,chunck_size,search_range,number_of_evaluations):
        # function for setting up the search chunck
        search_index_list = []
        
        # defines the search range of each chunck
        self.search_range = chunk_size*search_range/number_of_evaluations

        # iterate through all indicies to set up an index array 
        for i in range(number_of_evaluations):
            for j in range(number_of_evaluations):
                for k in range(number_of_evaluations):
                    search_index_list.append([-search_range+self.search*(k*2+1),
                                              -search_range+self.search*(j*2+1),
                                              -search_range+self.search*(i*2+1)])
        # transform list into array
        self.search_index_array = np.array(search_index_list)

        return self.search_index_array, self.search_range

    def worker(self,work_package):
        # brute force optimisation
        brute_force_output = rotate(model_1=self.model1,
                                    model_2=self.model2,
                                    number_of_evaluations=self.number_of_evaluations,
                                    full_output=True,
                                    order_spline_interpolation=self.order_spline_interpolation,
                                    cropping_model=self.cropping_model,
                                    mask=self.mask,
                                    method='brute_force',
                                    radius_radial_mask=self.radius_radial_mask,
                                    search_range=work_package['search_range'],
                                    log_model=True)

        # transforming the output to an array
        brute_force_output = np.array(brute_force_output)
        error_matrix = brute_force_output[2]

        # create a output dictionary
        res = {'error_matrix':error_matrix,
               'search_chunck_range':work_package['search_range']}

        return res


    def get_work(self):
        # iterate through the number of tasks
        if self.counter < self.n_tasks:
            self.counter += 1

            # create a dictionary
            work_package = {'search_index':self.search_index_array[self.counter],
                            'search_range':self.search_range}
            return work_package

        else:
            return None


    def logres(res):
        #pid = os.getpid()
        #with h5py.File(output_directory + '%s'%(pid) + '.h5', 'w') as f:
            #f['error_matrix'] = res['error_matrix'][:]
            #f['search_chunck_range'] = res['search_chunck_range'][:]
        return res
            
    def run(self):
        mulpro(Nprocesses=self.number_of_processes, worker=worker(), getwork=get_work, logres=logres)

"""
def main(model1_filename,model2_filename,model1_dataset,model2_dataset,
         number_of_processes=1,chunck_size=10,number_of_evaluations=10,
         order_spline_interpolation=3,cropping_model=None,mask=None,
         radius_radial_mask=None,search_range=np.pi/2.):

    get_error_matrix = ErrorMatrixBruteForce(model1_filename,
                                             model2_filename,
                                             model1_dataset,
                                             model2_dataset,
                                             number_of_processes=1,
                                             chunck_size=10,
                                             number_of_evaluations=10,
                                             order_spline_interpolation=3,
                                             cropping_model=None,
                                             mask=None,
                                             radius_radial_mask=None,
                                             search_range=np.pi/2.)
    get_error_matrix.run()

if __name__ == '__main__':
    main()
"""
