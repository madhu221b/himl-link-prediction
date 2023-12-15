import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from collections import Counter
from utils import get_fraction_in_topk, get_centrality

seeds = [42,420,4200]

def get_visibility_dict(folder_path, hMM,hmm):
    visibility_dict = {}
    all_files = os.listdir(folder_path)
    csv_files = [os.path.join(folder_path,file_name) for file_name in all_files if "netmeta" not in file_name and ".csv" in file_name]
    for file_name in csv_files:
        hMM_ext, hmm_ext = file_name.split("hMM")[-1].split("-")[0], file_name.split("hmm")[-1].split("-")[0]
        hMM_ext, hmm_ext = float(hMM_ext.replace(".csv","")), float(hmm_ext.replace(".csv",""))
        if hmm_ext == hmm and hMM_ext == hMM:
            T =  int(file_name.split("n_epoch_")[-1].replace(".csv",""))
            fm_hat = get_fraction_in_topk(file_name)
            visibility_dict[T] = fm_hat
   
    visibility_dict = {key:val-visibility_dict[0] for key,val in visibility_dict.items()}
    visibility_dict = dict(sorted(visibility_dict.items()))
    return visibility_dict
            
def get_centrality_plot(hmm, hMM,B, no_human=False,centrality="betweenness"):
    if no_human: model = "_no_human/DPAH"
    else: model = "_human/B_{}/DPAH".format(B)
    folder_path = "../himl-link-prediction/{}/".format(model)
    centrality_dict = {}
    all_files = os.listdir(folder_path)
    gpkl_files = [os.path.join(folder_path,file_name) for file_name in all_files if "csv" not in file_name and ".gpickle" in file_name]
    
    dict_folder = "./centrality/{}/{}".format(centrality,model)
    if not os.path.exists(dict_folder): os.makedirs(dict_folder)
    dict_file_name = dict_folder+"/_hMM{}_hmm{}.pkl".format(hMM,hmm)

    if not os.path.exists(dict_file_name):
        for file_name in gpkl_files:
            hMM_ext, hmm_ext = file_name.split("hMM")[-1].split("-")[0], file_name.split("hmm")[-1].split("-")[0]
            hMM_ext, hmm_ext = float(hMM_ext.replace(".gpickle","")), float(hmm_ext.replace(".gpickle",""))
            if hmm_ext == hmm and hMM_ext == hMM:
                T =  int(file_name.split("n_epoch_")[-1].replace(".gpickle",""))
                avg_cent = get_centrality(file_name,centrality="betweenness")
                centrality_dict[T] = np.round(avg_cent,5)
        with open(dict_file_name, 'wb') as f:                
                pkl.dump(centrality_dict,f)

    else:
        with open(dict_file_name, 'rb') as f:                
            centrality_dict = pkl.load(f)

def time_vs_betn(hMM,hmm):
    colors = ["#E9EC2C","#2C92EC","#38A055"]
    no_human_path = "../himl-link-prediction/centrality/betweenness/_no_human/DPAH/_hMM{}_hmm{}.pkl".format(hMM,hmm)
    with open(no_human_path, 'rb') as f:                
            centrality_dict = pkl.load(f)
            centrality_dict = dict(sorted(centrality_dict.items()))

    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot(centrality_dict.keys(), centrality_dict.values(), label ='No Human', marker="o",color="#DD654B")
    idxs = list(centrality_dict.keys())

    human_path = "/home/mpawar/himl-link-prediction/centrality/betweenness/_human"
    for i, folder_path in enumerate(os.listdir(human_path)):
        step_size = folder_path.split("B_")[-1]
        dict_path = os.path.join(human_path,folder_path,"DPAH","_hMM{}_hmm{}.pkl".format(hMM,hmm))
        with open(dict_path, 'rb') as f:                
            centrality_dict = pkl.load(f)
            centrality_dict = dict(sorted(centrality_dict.items()))
            ax.plot(centrality_dict.keys(), centrality_dict.values(),marker="o",color=colors[i],label="AL, B={}".format(step_size))
    ax.set_xticks(idxs)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Avg Betweeness Centrality for Minority Nodes")
    ax.legend(loc = "lower right",bbox_to_anchor=(0.5,0))
    fig.savefig('plots/time_vs_betn.png',bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window


def get_whole_cent_dict(main_path,hMM,hmm):
    print(main_path)
    dict_file = main_path+"/centrality_hMM{}_hmm{}.pkl".format(hMM,hmm)
    cent_dict = {}
    if not os.path.exists(dict_file):
            for file_name in os.listdir(main_path):
              if ".gpickle" not in file_name or "csv" in file_name: continue 
              hMM_ext, hmm_ext = file_name.split("hMM")[-1].split("-")[0], file_name.split("hmm")[-1].split("-")[0]
              hMM_ext, hmm_ext = float(hMM_ext.replace(".gpickle","")), float(hmm_ext.replace(".gpickle",""))
          
              print("dict file", dict_file)
              if hMM != hMM_ext or hmm != hmm_ext: continue

              
              graph_path = os.path.join(main_path,file_name)
              print("graph path: ", graph_path)
              T =  int(graph_path.split("n_epoch_")[-1].replace(".gpickle",""))
              avg_cent = get_centrality(graph_path,centrality="betweenness")
              cent_dict[T] = np.round(avg_cent,5)

            with open(dict_file, 'wb') as f:                
                    pkl.dump(cent_dict,f)                 
    else:
            with open(dict_file, 'rb') as f:                
               cent_dict = pkl.load(f)
               cent_dict = dict(sorted(cent_dict.items()))
        
    cent_dict = dict(sorted(cent_dict.items()))
    return cent_dict


def time_vs_betn_seeds(hMM,hmm):
    colors = ["#81B622","#D8A7B1","#2C92EC","#38A055"]
    no_human_path, human_path = "../himl-link-prediction/_no_human/",  "../himl-link-prediction/_human/"
    no_human_dict, human_dict = {}, {}
    for i, seed in enumerate(seeds):
       
        seed_path = os.path.join(no_human_path, "seed_"+str(seed),"B_0/dim_64/DPAH")
        cent_dict = get_whole_cent_dict(seed_path,hMM,hmm)
        
        b_dict = {}
        # [50,75,100,200]
        for j, B in enumerate([75]):
            seed_path = os.path.join(human_path, "seed_"+str(seed),"B_{}/dim_64/DPAH".format(B))
            h_cent_dict = get_whole_cent_dict(seed_path,hMM,hmm)
            b_dict[B] = h_cent_dict
           

        if i == 0:
            no_human_dict = {key:[value] for key,value in cent_dict.items()}
            human_dict = {B:{T:[val] for T, val in sub_dict.items()}  for B,sub_dict in b_dict.items()} # {"B":{0,1,2...}}
           
        else: # add values
           no_human_dict = {key:no_human_dict[key]+[value] for key,value in cent_dict.items()}
           human_dict = {B:{T:human_dict[B][T]+[val] for T, val in sub_dict.items()}  for B,sub_dict in b_dict.items()}
           if i == len(seeds) -1:
              no_h_std = [np.std(value) for _,value in no_human_dict.items()]
              no_human_dict = {key:(np.round(np.mean(value),5)) for key,value in no_human_dict.items()}
              
              h_std =  [np.std(val) for B,sub_dict in human_dict.items() for T, val in sub_dict.items()]
              print("no h st:", no_h_std)
              print("no humn :", no_human_dict)
         
              human_dict = {B:{T:(np.round(np.mean(val),5)) for T, val in sub_dict.items()}  for B,sub_dict in human_dict.items()}
              print("h st: ", h_std)
              print("humn :",human_dict)


    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    # ax.plot(no_human_dict.keys(), no_human_dict.values(), label ='No Human', marker="o",color="#DD654B")
  

    idxs = list(no_human_dict.keys())

    
    for i, (b, sub_dict) in enumerate(human_dict.items()):
       ax.errorbar(sub_dict.keys(), sub_dict.values(), label="AL, B={}".format(b), marker="o",color=colors[i],yerr = h_std)
    

    ax.errorbar(no_human_dict.keys(), no_human_dict.values(), label ='No Human', marker="o",color="#DD654B", yerr = no_h_std)
    ax.set_xticks(idxs)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Avg Betweeness Centrality for Minority Nodes")
    ax.legend(loc = "lower right",bbox_to_anchor=(0.4,0))
    fig.savefig('plots/time_vs_betn_hMM{}_hmm{}.png'.format(hMM,hmm),bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window

def time_vs_visibility_seeds(hMM,hmm):
    colors = ["#81B622","#D8A7B1","#2C92EC","#38A055"]
    no_human_path, human_path = "../himl-link-prediction/_no_human/",  "../himl-link-prediction/_human/"
    no_human_dict, human_dict = {}, {}
    for i, seed in enumerate(seeds):
       
        seed_path = os.path.join(no_human_path, "seed_"+str(seed),"B_0/dim_64/DPAH")
        vis_dict = get_visibility_dict(seed_path,hMM,hmm)
        
        b_dict = {}
        # [50,75,100,200]
        for j, B in enumerate([75]):
            seed_path = os.path.join(human_path, "seed_"+str(seed),"B_{}/dim_64/DPAH".format(B))
            vis_dict_2 = get_visibility_dict(seed_path,hMM,hmm)
            b_dict[B] = vis_dict_2
           

        if i == 0:
            no_human_dict = {key:[value] for key,value in vis_dict.items()}
            human_dict = {B:{T:[val] for T, val in sub_dict.items()}  for B,sub_dict in b_dict.items()} # {"B":{0,1,2...}}
           
        else: # add values
           no_human_dict = {key:no_human_dict[key]+[value] for key,value in vis_dict.items()}
           human_dict = {B:{T:human_dict[B][T]+[val] for T, val in sub_dict.items()}  for B,sub_dict in b_dict.items()}
           if i == len(seeds) -1:
              no_h_std = [np.std(value) for _,value in no_human_dict.items()]
              no_human_dict = {key:(np.round(np.mean(value),5)) for key,value in no_human_dict.items()}
              
              h_std =  [np.std(val) for B,sub_dict in human_dict.items() for T, val in sub_dict.items()]
              print("no h st:", no_h_std)
              print("no humn :", no_human_dict)
         
              human_dict = {B:{T:(np.round(np.mean(val),5)) for T, val in sub_dict.items()}  for B,sub_dict in human_dict.items()}
              print("h st: ", h_std)
              print("humn :",human_dict)


    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    # ax.plot(no_human_dict.keys(), no_human_dict.values(), label ='No Human', marker="o",color="#DD654B")
  

    idxs = list(no_human_dict.keys())

    
    for i, (b, sub_dict) in enumerate(human_dict.items()):
       ax.errorbar(sub_dict.keys(), sub_dict.values(), label="AL, B={}".format(b), marker="o",color=colors[i],yerr = h_std)
    

    ax.errorbar(no_human_dict.keys(), no_human_dict.values(), label ='No Human', marker="o",color="#DD654B", yerr = no_h_std)
    ax.set_xticks(idxs)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Visibility of Minority Nodes in Top-10")
    ax.legend(loc = "lower right",bbox_to_anchor=(0.8,0))
    fig.savefig('plots/time_vs_visibility_hMM_{}_hmm_{}.png'.format(hMM,hmm),bbox_inches='tight') 
    plt.close(fig)    # close the figure window


def time_vs_visibility(hMM,hmm):
    colors = ["#E9EC2C","#2C92EC","#38A055"]         
    visibility_dict = get_visibility_plot(hmm, hMM, no_human=True)
    visibility_dict = dict(sorted(visibility_dict.items()))

    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot(visibility_dict.keys(), visibility_dict.values(), label ='No Human',color="#DD654B")
    idxs = list(visibility_dict.keys())


    for i, B in enumerate([75]):          
        visibility_dict = get_visibility_plot(hmm, hMM, B=B)
        visibility_dict = dict(sorted(visibility_dict.items()))
        ax.plot(visibility_dict.keys(), visibility_dict.values(),color=colors[i],label="AL, B={}".format(B))
    ax.set_xticks(idxs)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Change in Visibility of Minority Nodes in Top-10")
    ax.legend(loc = "lower right",bbox_to_anchor=(0.8,0))
    fig.savefig('plots/time_vs_visibility_hMM_{}_hmm_{}.png'.format(hMM,hmm),bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window


def time_vs_betn_dims():
    path = "../himl-link-prediction/_human/B_75"
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    colors = ["#DD654B","#E9EC2C","#2C92EC","#38A055"] 

    for i, dim in enumerate(os.listdir(path)):
        if not "dim" in dim: continue
        print("Dimensionality: ", dim)
        sub_path = os.path.join(path,dim,"DPAH")
        cent_dict, dict_file = {}, sub_path+"/centrality.pkl".format(dim)
        if not os.path.exists(dict_file):
            for file_name in os.listdir(sub_path):
              if ".gpickle" not in file_name or "csv" in file_name: continue 
              graph_path = os.path.join(sub_path,file_name)
             
            
              T =  int(graph_path.split("n_epoch_")[-1].replace(".gpickle",""))
              avg_cent = get_centrality(graph_path,centrality="betweenness")
              cent_dict[T] = np.round(avg_cent,5)

            with open(dict_file, 'wb') as f:                
                    pkl.dump(cent_dict,f)                 
        else:
            with open(dict_file, 'rb') as f:                
               cent_dict = pkl.load(f)
               idxs = list(cent_dict.keys())
               cent_dict = dict(sorted(cent_dict.items()))
               print("Centrality Dict computed for dim:{}, {}".format(dim,cent_dict))
               dim = dim.split("dim_")[-1]
               ax.plot(cent_dict.keys(), cent_dict.values(),marker="o",color=colors[i-1],label="dim={}".format(dim))

                

    
    ax.set_xticks(idxs)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Avg Betweeness Centrality for Minority Nodes")
    ax.legend(loc = "lower right",bbox_to_anchor=(0.4,0))
    fig.savefig('plots/time_vs_cent_b=75_dims.png',bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window



if __name__ == "__main__":
    # time_vs_betn_dims()
    hMM, hmm = 0.5, 0.5
    time_vs_visibility_seeds(hMM,hmm)
    # time_vs_betn_seeds(hMM,hmm)