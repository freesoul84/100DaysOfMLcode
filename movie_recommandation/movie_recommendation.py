#importing all libraries
import numpy as np
import pandas as pd
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetching of movies having rating 5
m_u_data = fetch_movielens(min_rating=5.0)
print(m_u_data)
print(repr(m_u_data['train']))
print(repr(m_u_data['test']))
print(m_u_data['train'])
print(m_u_data['item_labels'])

#model creation
model = LightFM(loss='warp')

#fiting of a model with epoch 40
model.fit(m_u_data['train'], epochs=40, num_threads=2)

def recommendation(model,m_u_data,uids):
            n_users= m_u_data['train'].shape[0]
            n_items=m_u_data['train'].shape[1]
            for uid in uids:
                        known_positive =m_u_data['item_labels'][m_u_data['train'].tocsr()[uid].indices]
                        scores = model.predict(uid, np.arange(n_items))
                        top_items = m_u_data['item_labels'][np.argsort(-scores)]
                        print("User :",uid)
                        print("known movies :")
                        for x in known_positive[:3]:
                              print(x)
                        print("\t")
                       
                        print("Recomended for userid %s"%uid)
                        for x in top_items[:3]:
                              print(x)
                        print("**************************************")

recommendation(model, m_u_data, [5,6])
