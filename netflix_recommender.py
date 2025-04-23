# Netflix-Style Movie Recommendation Engine using Keras & TensorFlow

# 📁 Load Dataset (make sure your CSV has 'userId', 'movieId', 'rating')

df = pd.read_csv("/content/drive/MyDrive/Movies datasets/movies.csv")  # Replace with your movie CSV file

# 🧹 Preprocess data
le_user = LabelEncoder()
df['user'] = le_user.fit_transform(df['id'])
le_movie = LabelEncoder()
df['movie'] = le_movie.fit_transform(df['original_title'])

# ✂️ Split data into train/test
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# 📐 Define model parameters
num_users = df['user'].nunique()
num_movies = df['movie'].nunique()
embedding_size = 50

# 🎯 Inputs
user_input = Input(shape=(1,), name='user')
movie_input = Input(shape=(1,), name='movie')

# 🔗 Embeddings
user_embedding = Embedding(num_users, embedding_size, name='user_embedding')(user_input)
movie_embedding = Embedding(num_movies, embedding_size, name='movie_embedding')(movie_input)

# ✨ Dot product + flatten
dot_product = Dot(axes=-1)([user_embedding, movie_embedding])
flatten = Flatten()(dot_product)

# 🧠 Prediction layer
output = Dense(1)(flatten)

# 🏗️ Build and compile model
model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(optimizer='adam', loss='mse')
model.summary()

# 📊 Prepare data for training
X_train = train_data[['user', 'movie']]
y_train = train_data['rating']
X_test = test_data[['user', 'movie']]
y_test = test_data['rating']

# 🏃‍♂️ Train the model
model.fit([X_train['user'], X_train['movie']], y_train,
          validation_data=([X_test['user'], X_test['movie']], y_test),
          epochs=10, batch_size=64, verbose=1)

# 📈 Evaluation using RMSE
preds = model.predict([X_test['user'], X_test['movie']])
rmse = math.sqrt(mean_squared_error(y_test, preds))
print(f"RMSE: {rmse}")

# 🎁 Recommendation function
def recommend_movies(user_id, df_original, model, top_n=10):
    user_index = le_user.transform([user_id])[0]
    all_movies = df_original['movieId'].unique()
    encoded_movies = le_movie.transform(all_movies)
    user_array = np.array([user_index for _ in range(len(encoded_movies))])
    
    predictions = model.predict([user_array, encoded_movies])
    recommended_ids = encoded_movies[np.argsort(predictions.flatten())[::-1][:top_n]]
    recommended_movieIds = le_movie.inverse_transform(recommended_ids)
    
    return df_original[df_original['id'].isin(recommended_movieIds)][['original_language', 'original_title']].drop_duplicates()

# 🔍 Example usage
# print(recommend_movies(user_id=1, df_original=df, model=model, top_n=5))
