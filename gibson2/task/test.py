import pickle as pkl 

with open ('../ig_dataset/objects/fridge/11211/placements/shelf_0/processed_for_sampling/shelf_setup_0.pkl', 'rb') as f:
    A = pkl.load(f)
print(A)