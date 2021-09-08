import os
import yagmail
from dotenv import load_dotenv

user_name = os.getenv('EMAIL') #email of the gmail account you want to send emails from

def email_send(to,message):
  
  html = """\
  <html>
    <body>
      <h1>Thank you for contacting me<h1>
      <h3>This is a computer generated email, I will get in touch regarding your query as soon as possible.</h3>
      <br>
      <br>
      <p><a href="https://github.com/Rishabh1501">Github Profile</a><br>
      <a href="https://www.linkedin.com/in/rishabh-kalra-87ab151b2/">Linkedin Profile</a></p>
    </body>
  </html>"""
  
  yag = yagmail.SMTP(user_name)
  yag.send(
      to=to,
      subject="Query Regarding IMDB Review Analysis",
      contents = html
  )
  yag.send(
      to=os.getenv("TO"), 
      subject=f"Query by {to}",
      contents = message
  )  #to = your personal email id on which you want to recieve customer queries