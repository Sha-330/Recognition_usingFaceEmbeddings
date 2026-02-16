import json
from scipy.spatial.distance import cosine
from deepface import DeepFace
import cv2 # Required by DeepFace
import numpy as np # Import numpy for handling potential numpy arrays from loaded embeddings

# Define the filename for embeddings
embeddings_filename = "face_embeddings.json"

# Load the dictionary from the JSON file
try:
    with open(embeddings_filename, 'r') as f:
        face_embeddings_dict = json.load(f)
    print(f"Face embeddings loaded from {embeddings_filename}")
    # Convert list embeddings back to numpy arrays if necessary for cosine distance
    for name, embedding_list in face_embeddings_dict.items():
        face_embeddings_dict[name] = np.array(embedding_list)
except FileNotFoundError:
    print(f"Error: {embeddings_filename} not found. Please ensure the embeddings file exists.")
    face_embeddings_dict = {} # Initialize as empty dict if file not found
except Exception as e:
    print(f"An error occurred while loading embeddings: {e}")
    face_embeddings_dict = {} # Initialize as empty dict in case of other errors


# Load the test image
test_image_path = r"C:\Users\mazin\Pictures\Screenshots\Screenshot 2025-10-15 110811.png"

try:
    # Attempt to extract embeddings from the test image
    test_embeddings = DeepFace.represent(img_path = test_image_path, model_name ="VGG-Face", detector_backend = "opencv")

    if test_embeddings:
        print(f"Successfully extracted embeddings for {len(test_embeddings)} face(s) in the test image.")
        # For simplicity, we will compare the first detected face in the test image
        test_embedding = test_embeddings[0]['embedding']

        # --- Comparison with loaded embeddings ---
        if face_embeddings_dict and test_embedding is not None:
            # Iterate through known embeddings and find the closest match
            closest_match = None
            min_distance = float('inf')

            for name, known_embedding in face_embeddings_dict.items():
                # Ensure both embeddings are numpy arrays for cosine calculation
                if isinstance(known_embedding, list):
                     known_embedding = np.array(known_embedding)
                if isinstance(test_embedding, list):
                    test_embedding = np.array(test_embedding)

                distance = cosine(known_embedding, test_embedding)
                if distance < min_distance:
                    min_distance = distance
                    closest_match = name

            # You can set a threshold for recognition. A common threshold for VGG-Face is around 0.68.
            threshold = 0.68

            if closest_match and min_distance < threshold:
                print(f"The person in the test image is likely {closest_match} (distance: {min_distance:.4f})")
            elif closest_match:
                 print(f"The person in the test image is likely NOT {closest_match} (closest distance found: {min_distance:.4f})")
            else:
                 print("No known faces to compare against.")
        else:
            print("Could not perform comparison. No known embeddings loaded or no face detected in the test image.")

    else:
        print("No faces detected in the test image.")


except Exception as e:
    print(f"An error occurred during test image processing or comparison: {e}")
