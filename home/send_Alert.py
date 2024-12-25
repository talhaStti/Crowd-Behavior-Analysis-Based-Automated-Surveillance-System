from email.message import EmailMessage
from email.mime.image import MIMEImage
import ssl
import smtplib
import winsound
import cv2


def send_alert(user_email):
    print('sending email')
    email_senders = 'smartsurviellance@gmail.com'
    email_reciever = user_email
    email_password = 'cjnzzvnkopntvcfw'

    subject = 'ALERT'

    body = """"
Alert!!! Violence is detected on your premises. Please take the necessary action,
"""
    alert = EmailMessage()
    alert['From'] = email_senders
    alert['To'] = email_reciever
    alert['Subject'] = subject
    alert.set_content(body)
    # _, i1 = cv2.imencode('.jpg', f)
    # img = MIMEImage(i1.tobytes())
    # img.add_header('Content-Disposition', 'attachment',
    #                filename='snapshot.jpg')
    # alert.add_attachment(img)

    Context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=Context) as s:
        s.login(email_senders, email_password)
        s.sendmail(email_senders, email_reciever, alert.as_string())
        print("Sent email")
