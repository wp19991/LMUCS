from djitellopy import Tello

drone_data = {}

tello = Tello()
tello.connect()
tello.query_sdk_version()
drone_data['battery'] = tello.get_battery()
drone_data['udp_video_address'] = tello.get_udp_video_address()
print(drone_data)

tello.connect_to_wifi(ssid='x', password='88888888')
