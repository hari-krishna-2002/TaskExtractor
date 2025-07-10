# calendar_helper.py

import datetime
import pytz
from googleapiclient.discovery import build
from google.oauth2 import service_account

SERVICE_ACCOUNT_FILE = 'service_acount.json'  # Ensure correct path
SCOPES = ['https://www.googleapis.com/auth/calendar']
CALENDAR_ID = 'harikrishnakola1234@gmail.com'  # Replace with actual

def get_calendar_service():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        return build('calendar', 'v3', credentials=credentials)
    except Exception as e:
        print(f"❌ Failed to load credentials or create service: {e}")
        raise

def create_event(task_text, due_date, description=None):
    try:
        service = get_calendar_service()

        due_datetime = datetime.datetime.strptime(due_date, "%Y-%m-%d").replace(hour=9)
        timezone = pytz.timezone("Asia/Kolkata")
        due_datetime = timezone.localize(due_datetime)

        event = {
            'summary': task_text,
            'description': description or '',
            'start': {
                'dateTime': due_datetime.isoformat(),
                'timeZone': 'Asia/Kolkata',
            },
            'end': {
                'dateTime': (due_datetime + datetime.timedelta(hours=1)).isoformat(),
                'timeZone': 'Asia/Kolkata',
            },
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'popup', 'minutes': 10},
                    {'method': 'email', 'minutes': 30},
                ],
            },
        }

        event = service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
        print(f"✅ Event created: {event.get('htmlLink')}")

    except Exception as e:
        print(f"❌ Failed to create calendar event: {e}")
