import logging
from botocore.exceptions import ClientError
import json

logger = logging.getLogger(__name__)

# This utility function checks if a rekognition model is ready to use
def check_rekognition_model_status(client, project_arn, version_name):
    try:
        describe_response = client.describe_project_versions(
                        ProjectArn=project_arn,
                        VersionNames=[version_name])
        
        if len(describe_response['ProjectVersionDescriptions'])<1:
            logger.error("No Rekognition mode found...")
            return False
        
        #Get model status
        model = describe_response['ProjectVersionDescriptions'][0]
        status = model['Status']
        
        if status != "RUNNING":
            logger.error("Rekognition Model Not Running...")
            return False
    
    except ClientError as client_err:
        logger.error(format(client_err))
        return False
    
    return True


# This is a utility function to call rekognition model and run inference on a image
def analyze_video_frame(rek_client, model_arn, frame, min_confidence):

    try:

        response = rek_client.detect_custom_labels(Image={'Bytes': frame},
            MinConfidence=min_confidence,
            ProjectVersionArn=model_arn)
        
        return response['CustomLabels']

    except ClientError as client_err:
        logger.error(format(client_err))
        raise
        
# This function crops out a bbox from an image
def crop_images(frame, bbox):
    
    bbox_new = dict()
    
    height = frame.shape[0]
    width = frame.shape[1]

    bbox_w = int(bbox['Width']*width)
    bbox_h = int(bbox['Height']*height)
    bbox_l = int(bbox['Left']*width)
    bbox_t = int(bbox['Top']*height)
    
    bbox_new['top'] = bbox_t
    
    bbox_new['bottom'] = bbox_t+bbox_h
    
    bbox_new['left'] = bbox_l
    
    bbox_new['right'] = bbox_l+bbox_w

    cropped = frame[bbox_t:bbox_t+bbox_h, bbox_l:bbox_l+bbox_w]
    return cropped, bbox_new

# This function extracts the embedding results from a Sagemaker endpoint
def parse_response(query_response):
    """Parse response and return the embedding."""

    model_predictions = json.loads(query_response)
    translation_text = model_predictions["embedding"]
    return translation_text

# This function calls a sagemaker endpoint to generate embedding
def generate_embedding(img_bytes, model_predictor):
    
    query_response = model_predictor.predict(
        img_bytes,
        {
            "ContentType": "application/x-image",
            "Accept": "application/json",
        },
    )
    
    vector = parse_response(query_response)
    return vector

# This function calls opensearch to get knn search results
def knn_search(opensearch, index_name, k, embedding_vector):
    query = {
        "size":k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": embedding_vector,
                    "k": k
                }
            }
        }
    }
    
    res = opensearch.search(index = index_name, body=query)

    results = []
    for r in res["hits"]["hits"]:
        results.append([r["_score"], r["_source"]["product_id"], r["_source"]["product_category"]])
    
    return results

#This function draws bbox on images
def draw_bbox(cv2, frame, bbox):
    
    top = bbox['top']
    left = bbox['left']
    bottom = bbox['bottom']
    right = bbox['right']
    
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 165, 255), 10)
    
    return frame
    
    