## Email Spam Detector

## DESCRIPTION

"Email Spam Detector" is a serverless, microservice driven application, powered by a machine learning model to predict whether an email is spam or not. This system that upon receipt of an email message, automatically flag it as spam or not, based on the prediction obtained from the ML model. And it replies back to the sender of the email with a reasonable prediction. It is designed using multiple AWS components :-
#### Amazon Sagemaker, AWS S3, AWS Lambda and AWS SES

## ARCHITECHTURE :- 
![alt text](https://github.com/im-vaibhav/Intelligent-System-Design-using-Cloud/blob/master/images/SpamEndpoint-designer.png)

## SAMPLE OUTPUT 
### Known Visitor 
The sender recieves an email with reasonable predictions about spam/not spam
![alt text](https://github.com/im-vaibhav/Intelligent-System-Design-using-Cloud/blob/master/images/spam_notify_email.png)



