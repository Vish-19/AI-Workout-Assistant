import cv2
from workout import dumbell_curl_simultanious
cap = cv2.VideoCapture('samples/dumbell_curls_simultanious/sample1.mp4')
dumbell_curl_simultanious(cap)
