import time
import logging
import argparse

import cv2
import boto3
from botocore.exceptions import ClientError
from sagemaker.predictor import Predictor
import sagemaker, json
from sagemaker import get_execution_role
from os import listdir

from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

import octank_utility as utility

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

#initialize general info
input_path = "/opt/ml/processing/input"
output_path = '/opt/ml/processing/output'

print(listdir(input_path))

region = 'us-east-1'
sess = sagemaker.Session(boto3.session.Session(region_name=region))


#initialize rekognition service
rekognition = boto3.client('rekognition', region_name=region)

project_arn='arn:aws:rekognition:us-east-1:913089978341:project/retail-test-2/1648085376750'
model_arn='arn:aws:rekognition:us-east-1:913089978341:project/retail-test-2/'+\
            'version/retail-test-2.2022-03-23T20.40.12/1648086013117'

version_name='retail-test-2.2022-03-23T20.40.12'

# initialize SageMaker endpoint
embedding_endpoint = 'jumpstart-example-infer-tensorflow-icem-2022-04-14-15-14-03-830'
# embedding model predictor
model_predictor = Predictor(endpoint_name = embedding_endpoint,
                           sagemaker_session = sess)

#initialize opensearch
opensearch_domain_endpoint = 'search-image-search-2-5citm4637xvf5ufci35pbb6g4u.us-east-1.es.amazonaws.com'

index_name = 'image_embedding'

credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, region)

opensearch_host = {
    'host' : opensearch_domain_endpoint,
    'port' : 443,
    'scheme' : 'https',
}

opensearch = OpenSearch(hosts = [opensearch_host],
               http_auth = auth,
                use_ssl = True,
                verify_certs = True,
                connection_class = RequestsHttpConnection)

#initialize DynamoDB
table_name = 'octank_movie'
dynamodb = boto3.resource('dynamodb', region_name=region)
table = dynamodb.Table(table_name)

def process_frames(frame):
    
    # get width and height info for each frame
    height = frame.shape[0]
    width = frame.shape[1]
    
    # send frame to Rekognition for inference
    logger.debug("Run product detection =============")

    image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
    results = utility.analyze_video_frame(rekognition, 
                                 model_arn, 
                                image_bytes,
                                80)
    
    detections = []
    for product in results:
        
        detect =dict()

        # crop images
        logger.debug("Crop detected product =============")
        cropped, bbox = utility.crop_images(frame, product['Geometry']['BoundingBox'])
    
        detect['bbox']=bbox

        cropped_bytes = cv2.imencode('.jpeg', cropped)[1].tobytes()

        # generate embedding
        logger.debug("Generate embedding =============")
        embedding_vector = utility.generate_embedding(cropped_bytes, model_predictor)
        
        # search product info from OpenSearch
        logger.debug("Search Query =============")
        results = utility.knn_search(opensearch, index_name, 1, embedding_vector)
        
        detect['product_id'] = results[0][1]
        
        detections.append(detect)
    
    
    return detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--start", type=int, default=0)
    args, _ = parser.parse_known_args()

    
    # check if rekognition model is ready
    ready = utility.check_rekognition_model_status(rekognition,
                                                   project_arn,
                                                   version_name)
    
    if ready:
        output = dict()
        
        output['video_id'] = args.filename
        
        video_path = f"{input_path}/{args.filename}"
        
        print(video_path)
        output['detections'] = dict()
        
        # load the movie video
        cap = cv2.VideoCapture(video_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total frame count: {frame_count} =============" )

        
        frame_id = args.start
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        
        try:
            while True:
                logger.debug(f"processing frame number {frame_id} ============")
                _, frame = cap.read()

                detections = process_frames(frame)

                for d in detections:
                    frame = utility.draw_bbox(cv2, frame, d['bbox'])

                output['detections'][str(frame_id)] = detections
                frame_id += 1
                if frame_id >= frame_count:
                    break
                    
            logger.debug("Upload output data to DynamoDB ==========")
            table.put_item(
                Item=output
            )
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
        
    logger.info("Finished running processing job")
