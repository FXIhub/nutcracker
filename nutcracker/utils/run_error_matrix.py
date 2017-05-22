from ErrorMatrixBruteForce import run

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
