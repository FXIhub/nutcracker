import numpy as np
from mulpro import mulpro
import os
import h5py
import nutcracker

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

        # define global variables
        self.number_of_evaluations = number_of_evaluations
        self.number_of_processes = number_of_processes
        self.chunck_size = chunck_size
        self.order_spline_interpolation = order_spline_interpolation
        self.cropping_model = cropping_model
        self.mask = mask
        self.radius_radial_mask = radius_radial_mask
        self.search_range = search_range

        self.counter = 0
        self.n_tasks = self.number_of_evaluations**3 / self.chunck_size**3
        self.n_evaluations = self.number_of_evaluations / self.chunck_size
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

    def get_chunck(self):
        # function for setting up the search chunck
        search_index_list = []
        
        # defines the search range of each chunck
        self.search_range_chunck = self.chunck_size*self.search_range/self.number_of_evaluations

        # iterate through all indicies to set up an index array 
        for i in range(self.n_evaluations):
            for j in range(self.n_evaluations):
                for k in range(self.n_evaluations):
                    search_index_list.append([-self.search_range+self.search_range_chunck*(k*2+1),
                                              -self.search_range+self.search_range_chunck*(j*2+1),
                                              -self.search_range+self.search_range_chunck*(i*2+1)])
        # transform list into array
        self.search_index_array = np.array(search_index_list)

        return self.search_index_array, self.search_range

    def get_work(self):
        self.get_chunck()
        # iterate through the number of tasks
        if self.counter < self.n_tasks:
            
            # create a dictionary
            work_package = {'search_index':self.search_index_array[self.counter],
                            'search_range':self.search_range_chunck}
            
            self.counter += 1


        else:
            work_package = None

        return work_package

    def worker(self, work_package):
        # brute force optimisation
        brute_force_output = nutcracker.utils.rotate.find_rotation_between_two_models(model_1=self.model1,
                                                                                      model_2=self.model2,
                                                                                      number_of_evaluations=self.number_of_evaluations,
                                                                                      full_output=True,
                                                                                      order_spline_interpolation=self.order_spline_interpolation,
                                                                                      cropping_model=self.cropping_model,
                                                                                      mask=self.mask,
                                                                                      method='brute_force',
                                                                                      radius_radial_mask=self.radius_radial_mask,
                                                                                      initial_guess=work_package['search_index'],
                                                                                      search_range=work_package['search_range'],
                                                                                      log_model=True)

        # extracting the error matrix
        error_matrix = brute_force_output['rotation_grid'][:]

        # create a output dictionary
        res = {'error_matrix':error_matrix,
               'search_chunck_range':work_package['search_range'],
               'search_index':work_package['search_index']}

        return res

    def logres(self,res):

        self.error_matrix.append(res['error_matrix'])
        self.index.append(res['search_index'])
        self.range.append(res['search_chunck_range'])

        self.results = {'error_matrix':self.error_matrix,
                        'index':self.index,
                        'chunck_range':self.range}
            
    def run(self):
        self.error_matrix = []
        self.index = []
        self.range = []

        mulpro(Nprocesses=self.number_of_processes, worker=self.worker, getwork=self.get_work, logres=self.logres)

        return self.results
