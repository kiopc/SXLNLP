import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_item_similarity(ratings_matrix):
    """计算物品的相似度矩阵"""
    # 转置矩阵使得物品为行，用户为列
    item_user_matrix = ratings_matrix.T
    similarity_matrix = cosine_similarity(item_user_matrix)
    return similarity_matrix

def predict_ratings(user_id, ratings_matrix, item_similarity_matrix):
    """根据物品相似度矩阵预测用户对每个物品的评分"""
    user_ratings = ratings_matrix[user_id]
    weighted_sum = np.dot(item_similarity_matrix, user_ratings)
    similarity_sum = np.array([np.abs(item_similarity_matrix[i]).dot((user_ratings != 0)) for i in range(len(item_similarity_matrix))])
    predictions = weighted_sum / (similarity_sum + 1e-8)  # 避免除零
    return predictions

# 示例评分矩阵 (用户-物品)
# 行表示用户，列表示物品，值为用户对物品的评分（0表示未评分）
ratings_matrix = np.array([
    [4, 0, 0, 5, 1, 0, 0],
    [5, 5, 4, 0, 0, 0, 0],
    [0, 0, 0, 2, 4, 5, 0],
    [0, 3, 0, 0, 0, 0, 3],
])

# 计算物品相似度矩阵
item_similarity_matrix = calculate_item_similarity(ratings_matrix)
print("物品相似度矩阵:\n", item_similarity_matrix)

# 为指定用户预测评分
user_id = 0  # 示例用户索引
predicted_ratings = predict_ratings(user_id, ratings_matrix, item_similarity_matrix)
print(f"用户 {user_id} 的预测评分:\n", predicted_ratings)
