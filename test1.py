import numpy as np
import pandas as pd
def Simple_filter(a:pd.DataFrame) -> pd.DataFrame:
        first_filter = a[(a["iptm"] > 0.8) & (a["ptm"] >= 0.7) 
                         & (a["has_clash"] == 0 ) & (a["fraction_disordered"] < 0.15)]
        first_filter.to_csv("first_filter.csv",encoding="utf-8-sig",index= False)
        c,d = first_filter.shape
        print(f"So hang {c}, So cot: {d}")
        return first_filter
        
data_frame = pd.read_csv("af3_candidates_10000_rows.csv")
a,b = data_frame.shape
print(f"So hang: {a} , So Cot: {b}")
print(data_frame.info())
first_layout = pd.read_csv("first_filter.csv")
print(first_layout.shape)
print(first_layout.head(3))
top_3_candidate = first_layout.sort_values(by = ["ranking_score"],ascending= False)
top_3_candidate.to_csv("sorted_list_by_ranking_score.csv",encoding= "utf-8-sig")
score = first_layout["ranking_score"].mean()
print(score)