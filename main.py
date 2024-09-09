import os
import torch
import clip
from PIL import Image
import numpy as np
import time
import pickle

class SimplePhotoSearchSystem:
    def __init__(self, photo_directory, embeddings_path='photo_embeddings.pkl', model_path='clip_model_state.pt'):
        self.photo_directory = photo_directory
        self.embeddings_path = embeddings_path
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.photo_embeddings = {}
        self.load_model()

    def load_model(self):
        print("Loading CLIP model...")
        start_time = time.time()
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        if os.path.exists(self.model_path):
            print(f"Loading model state from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print("Using default model weights")
            self.save_model()
        
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    def save_model(self):
        print(f"Saving model state to {self.model_path}")
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model state saved successfully")

    def load_embeddings(self):
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                self.photo_embeddings = pickle.load(f)
            print(f"Loaded {len(self.photo_embeddings)} embeddings from {self.embeddings_path}")
            return True
        return False

    def save_embeddings(self):
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.photo_embeddings, f)
        print(f"Saved {len(self.photo_embeddings)} embeddings to {self.embeddings_path}")

    def process_photos(self):
        existing_photos = set(self.photo_embeddings.keys())
        current_photos = set(f for f in os.listdir(self.photo_directory) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        
        new_photos = current_photos - existing_photos
        removed_photos = existing_photos - current_photos

        if new_photos or removed_photos:
            print(f"Detected {len(new_photos)} new photos and {len(removed_photos)} removed photos.")
            
            # Remove embeddings for deleted photos
            for photo in removed_photos:
                del self.photo_embeddings[photo]
            
            # Process new photos
            print("Processing new photos...")
            start_time = time.time()
            for filename in new_photos:
                image_path = os.path.join(self.photo_directory, filename)
                image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(image)
                self.photo_embeddings[filename] = image_features.cpu().numpy()
            
            print(f"Processed {len(new_photos)} new photos in {time.time() - start_time:.2f} seconds")
            self.save_embeddings()
        else:
            print("No new or removed photos detected.")

    def search_photos(self, query):
        print(f"Searching for '{query}'...")
        start_time = time.time()
        with torch.no_grad():
            text = clip.tokenize([query]).to(self.device)
            text_features = self.model.encode_text(text)

        similarities = {}
        for filename, embedding in self.photo_embeddings.items():
            similarity = np.dot(text_features.cpu().numpy(), embedding.T)
            similarities[filename] = similarity[0][0]

        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        print(f"Search completed in {time.time() - start_time:.2f} seconds")
        return sorted_results

    def get_top_result(self, query):
        results = self.search_photos(query)
        if not results:
            return None
        return results[0][0]  # Return just the filename of the top result

# Usage
print("Starting photo search system...")
overall_start_time = time.time()

search_system = SimplePhotoSearchSystem("photos", model_path="my_clip_model_state.pt")
search_system.load_embeddings()
search_system.process_photos()

queries = ["Cat","Dog", "Bear", "Person on beach with rugby ball"]
for query in queries:
    print(query)
    all_results = search_system.search_photos(query)
    print("\nAll results:")
    for filename, score in all_results:
        print(f"{filename}: {score:.4f}")

    top_result = search_system.get_top_result(query)
    print(f"\nTop result: {top_result}")

    print(f"\nTotal execution time: {time.time() - overall_start_time:.2f} seconds") git rev-list --objects --all | sort -k 2 -n | uniq | tail -n 10
