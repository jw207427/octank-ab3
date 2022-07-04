## Set up Personalize Recommendation engine

1. create personalize dataset using the 3 csv files: item, user, item-user-interaction
2. setup personalize permission to allow service to assume role and access data in s3 bucket
3. create recommenders: aws-ecomm-recommended-for-you, Customers Who Viewed X Also Viewed
4. Test recommenders
5. create event tracker and stream user interaction data to event tracker using Kinesis stream.