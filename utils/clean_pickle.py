import pickle
full_filename = "emmccpu_amb_2_bk.pickle"
a = pickle.load(open(full_filename,"rb"))
for k,v in a['modules'].items():
    v.interest_knn = None
with open(full_filename, 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
