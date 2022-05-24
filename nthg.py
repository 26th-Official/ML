import time
from twilio.rest import Client



time.sleep(180)
account_sid = 'AC744362365fd1c34cf5cc434c2ceb166b'
auth_token = 'b8a524034dd939edd98c6266b74727c9'
client = Client(account_sid, auth_token)


call = client.calls.create(
                        url='http://demo.twilio.com/docs/voice.xml',
                        to='+917358477623',
                        from_='+17755225005'
                    )

print(call.sid)


