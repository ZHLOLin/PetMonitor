import smtplib
import datetime
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

def send_mail(event, img_data=None):
    # Your Gmail account credentials
    gmail_user = 'linzhaolong.2010@gmail.com'
    gmail_password = 'fjirqrumrsagfjbd'

    # Email details
    from_email = gmail_user
    to_email = 'linzhaolong.2010@gmail.com'
    subject = 'Subject'
    if event == 0:
        now = datetime.datetime.now()
        body = 'Cat appears in the camera at' + now.strftime("%m/%d/%Y, %H:%M:%S")
    elif event == 1:
        now = datetime.datetime.now()
        body = 'Cat was detected moving fast' + now.strftime("%m/%d/%Y, %H:%M:%S")
    elif event == 2:
        now = datetime.datetime.now()
        body = 'Large noise deteced near' + now.strftime("%m/%d/%Y, %H:%M:%S")

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    if img_data is not None:
        image_mime = MIMEImage(img_data)
        msg.attach(image_mime)

    # Connect to Gmail server
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        # Send the email
        server.sendmail(from_email, to_email, msg.as_string())
        server.close()

        print('Email sent!')
    except Exception as e:
        print('Something went wrong:', e)