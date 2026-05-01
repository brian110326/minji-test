import numpy as np 
import config 

def cascade_fusion(p_search_results, c_vecs, p_vecs, embedding_db):
    final_results = []

    for p_res, c_v, p_v in zip(p_search_results, c_vecs, p_vecs):
     c_vec_1d = np.squeeze(c_v)  
     p_vec_1d = np.squeeze(p_v)
     refined_list = []

     # 1. 1차 필터링 결과 후보만 순회함
     for p_item in p_res:
        pid = p_item['paper_id']
        p_sim = p_item['score'] 

        target_vector = embedding_db.get(pid)
        if target_vector is None:
            continue

        # 2. context query 벡터와 후보 논문 벡터 간 유사도 계산 (코사인 유사도)
        c_sim = float(np.dot(c_vec_1d, target_vector))

        # 3. 가중합 (최종 점수)
        final_sim = config.PAPER_SIM_WEIGHT * p_sim + config.CONTEXT_SIM_WEIGHT * c_sim

        refined_list.append({
           "paper_id": pid,
            "sim": final_sim
        })

        # 4. 정렬 후 top-100 선발
        # 4. 정렬 후 Top-100 컷오프
        refined_list.sort(key=lambda x: x['sim'], reverse=True)
        top_100 = refined_list[:config.TOP_K_FINAL] # config에 100으로 설정되어 있다고 가정
        
        # 5. 결과 패키징
        placeholder_results = []
        q_id = p_res[0]['query_id'] if p_res else "UNKNOWN"
        for new_rank, cand in enumerate(top_100):
            placeholder_results.append({
                "query_id": q_id,
                "rank": new_rank + 1,
                "paper_id": cand['paper_id'],
                "sim": cand['sim']
            })
            
        final_results.append(placeholder_results)
        
    return final_results