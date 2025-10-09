import re

text = "192.168.1.1 - - [16/Sep/2025:10:30:00] \"GET /index.html HTTP/1.1\" 200 1024"
pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
print("IP: ",re.findall(pattern, text)[0])

pattern2 = r'\b\d{1,2}/\w{3}/\d{4}:\d{1,2}:\d{1,2}:\d{1,2}\b'
print("Fecha y hora: ",re.findall(pattern2, text)[0])

pattern3 = r'(?<=")[A-Z]{3,7}'
print("Método HTTP: ",re.findall(pattern3, text)[0])