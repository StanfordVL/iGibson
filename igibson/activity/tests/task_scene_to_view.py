import numpy as np

TASK_SCENE_TO_VIEW = {
	('chopping_vegetables', 'Rs_int'): [{
		'initial_pos': np.array([-0.80499908, 0.77876594, 1.5]),
		'initial_view_direction': np.array([ 7.01014331e-16, 9.63770896e-01, -2.66731437e-01]),
		'initial_up': np.array([0, 0, 1]),
	},
	{
	# countertop facing away from kitchen
		'initial_pos': np.array([-0.84676549,  2.22408238,  1.71      ]),
		'initial_view_direction': np.array([-3.65432517e-15, -9.78030915e-01, -2.08459900e-01]),
		'initial_up': np.array([0, 0, 1]),
	}],
	('re-shelving_library_books', 'Rs_int'): [{
		'initial_pos': np.array([0.61920862, 0.51876594, 1.16474328]),
		'initial_view_direction': np.array([ 9.63770896e-01, -5.37205886e-15, -2.66731437e-01]),
		'initial_up': np.array([0, 0, 1]),
	},
	{
	# reshelving books left
		'initial_pos': np.array([1.26263954, 1.21180578, 1.16474328]),
		'initial_view_direction': np.array([ 0.40730718, -0.87347307, -0.26673144]),
		'initial_up': np.array([0, 0, 1]),
	},
	{
	# reshelving books above
		'initial_pos': np.array([1.57689172, 0.294634  , 1.36756683]),
		'initial_view_direction': np.array([-0.29626606, -0.00296276, -0.95510086]),
		'initial_up': np.array([0, 0, 1]),
	},
	{
	# reshelving books right
		'initial_pos': np.array([ 1.46959988, -0.20658344,  1.16699565]),
		'initial_view_direction': np.array([-0.00936486,  0.86737765, -0.49756237]),
		'initial_up': np.array([0, 0, 1]),
	},
	{
	# reshelving books table 2
		'initial_pos': np.array([ 0.19354005, -3.24707663,  1.42573258]),
		'initial_view_direction': np.array([-4.32782095e-15, -8.57708681e-01, -5.14135992e-01]),
		'initial_up': np.array([0, 0, 1]),
	}],
	('storing_food', 'Rs_int'): [{
		'initial_pos': np.array([-0.79676549,  0.77408238,  1.71      ]),
		'initial_view_direction': np.array([ 7.24876323e-15,  9.78030915e-01, -2.08459900e-01]),
		'initial_up': np.array([0, 0, 1]),
	}],
	('assembling_gift_baskets', 'Rs_int'): [{
		'initial_pos': np.array([0.61920862, 0.51876594, 1.16474328]),
		'initial_view_direction': np.array([ 9.63770896e-01, -5.37205886e-15, -2.66731437e-01]),
		'initial_up': np.array([0, 0, 1]),
	}],
	('sorting_mail', 'Rs_int'): [{
		'initial_pos': np.array([0.04973646, 1.00102097, 1.56      ]),
		'initial_view_direction': np.array([ 0.00672076, -0.92488464, -0.38018842]),
		'initial_up': np.array([0, 0, 1]),
	}],
	('cleaning_up_the_kitchen_only', 'Rs_int'): [{
	    'initial_pos': np.array([1.01823427, 2.20146558, 1.5       ]),
        'initial_view_direction': np.array([-0.68924214,  0.01879785, -0.72428717]),
        'initial_up': np.array([0, 0, 1]),
	}],
	('polishing_silver', 'Rs_int'): [{
        'initial_pos': np.array([-1.7708582 ,  2.82369922,  0.63      ]),
        'initial_view_direction': np.array([6.123234e-17, 1.000000e+00, 0.000000e+00]),
        'initial_up': np.array([0, 0, 1]),
	}],
	# ('assembling_gift_baskets', 'Rs_int'): [{
	# 	'initial_pos': np.array([0.61920862, 0.51876594, 1.16474328]),
	# 	'initial_view_direction': np.array([ 9.63770896e-01, -5.37205886e-15, -2.66731437e-01]),
	# 	'initial_up': np.array([0, 0, 1]),
	# }],
	# Kitchen view
	 # 'initial_pos': np.array([-0.81,  0.7 ,  1.5 ]),
     #    'initial_view_direction': np.array([7.27366155e-16, 1.00000000e+00, 0.00000000e+00]),
     #    'initial_up': np.array([0, 0, 1]),
	### Pomaria_1_int
	('re-shelving_library_books', 'Pomaria_1_int'): [{
		'initial_pos': np.array([-1.03942999e+01,  5.20881680e-03,  1.16474328e+00]),
		'initial_view_direction': np.array([-9.63770896e-01,  2.11341517e-14, -2.66731437e-01]),
		'initial_up': np.array([0, 0, 1]),
	},
	{
	# reshelving books right
		'initial_pos': np.array([-11.25429986,   0.91655705,   0.94307597]),
		'initial_view_direction': np.array([ 1.06284760e-15, -9.63770896e-01, -2.66731437e-01]),
		'initial_up': np.array([0, 0, 1]),
	},
	{
	# reshelving books back
		'initial_pos': np.array([-1.21506220e+01,  1.00111862e-02,  9.43075974e-01]),
		'initial_view_direction': np.array([ 9.63770896e-01,  3.01078511e-14, -2.66731437e-01]),
		'initial_up': np.array([0, 0, 1]),
	},
	{
	# reshelving books left
		'initial_pos': np.array([-11.35081227,  -0.6443466 ,   0.94307597]),
		'initial_view_direction': np.array([-5.91528545e-14,  9.63770896e-01, -2.66731437e-01]),
		'initial_up': np.array([0, 0, 1]),
	}],
	('chopping_vegetables', 'Pomaria_1_int'): [{
		'initial_pos': np.array([ 0.04093782, -0.44312904,  1.85535483]),
		'initial_view_direction': np.array([ 4.63442178e-16,  6.37151144e-01, -7.70738879e-01]),
		'initial_up': np.array([0, 0, 1]),
	}],
}