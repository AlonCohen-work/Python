import cv2
import numpy as np

# קריאת התמונה
image = cv2.imread('image.jpg')

# שאלה 1
# מה עושה הקוד הבא, ומה התוצאה הצפויה?
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('Blurred', blurred)
cv2.waitKey(0)
# תשובה:
# שלב 1: התמונה הצבעונית מומרת לגווני אפור (gray).
# שלב 2: התמונה מטושטשת באמצעות Gaussian Blur עם מסנן בגודל 5x5 (blurred).
# שלב 3: התוצאה תוצג בחלון, והתמונה תהיה מטושטשת יותר כדי להפחית רעשים.

# הוספת קו אופקי אדום
height, width = image.shape[:2]
cv2.line(image, (0, height // 2), (width, height // 2), (0, 0, 255), 2)  # קו אדום

# הוספת קו אנכי כחול
cv2.line(image, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)  # קו כחול

# טשטוש רק לקוביה הימנית התחתונה
# קביעת גבולות לקוביה הימנית התחתונה
x_start = width // 2
x_end = width
y_start = height // 2
y_end = height

# חיתוך האזור הימני התחתון
bottom_right_region = image[y_start:y_end, x_start:x_end]

# טשטוש האזור
blurred_bottom_right = cv2.GaussianBlur(bottom_right_region, (15, 15), 0)

# החזרת האזור המטושטש לתמונה המקורית
image[y_start:y_end, x_start:x_end] = blurred_bottom_right

# הצגת התמונה הסופית עם הקווים והטשטוש
cv2.imshow('Final Image', image)
cv2.waitKey(0)

# שאלה 2
# שינוי גודל התמונה
resized = cv2.resize(image, (200, 200))
cv2.imshow('Resized Image', resized)
cv2.imwrite('resized_image.jpg', resized)  # יש לשמור את התמונה המוקטנת ולא את המקורית
cv2.waitKey(0)

# שאלה 3
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
cv2.imshow('Binary Image', binary)
cv2.waitKey(0)
# תשובה:
# שלב 1: התמונה מומרת לגווני אפור (gray).
# שלב 2: מבוצע סף בינארי על התמונה, כאשר פיקסלים מעל ערך 127 הופכים ל-255 (לבן), והאחרים הופכים ל-0 (שחור) (binary).
# שלב 3: התמונה הבינארית תוצג עם פיקסלים שחורים ולבנים בלבד.

# שאלה 4
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
cv2.imshow('Laplacian', laplacian)
cv2.waitKey(0)
# תשובה:
# שלב 1: התמונה מומרת לגווני אפור (gray).
# שלב 2: Gradient (שיפוע) של התמונה מחושב באמצעות Laplacian כדי לזהות שינויים חדים בעוצמת הפיקסלים (laplacian).
# שלב 3: השיפועים יוצגו בתמונה עם ערכים חיוביים ושליליים כפיקסלים בהירים וכהים.

# שאלה 5
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
inverted = cv2.bitwise_not(edges)
cv2.imshow('Inverted Edges', inverted)
cv2.waitKey(0)
# תשובה:
# שלב 1: התמונה מומרת לגווני אפור (gray).
# שלב 2: קווי מתאר מחושבים עם ספים 100 ו-200 (edges).
# שלב 3: קווי המתאר מתהפכים (הפיקסלים הלבנים הופכים לשחורים ולהפך) (inverted).
# שלב 4: התמונה תוצג עם קווי מתאר שחורים על רקע לבן.

# שאלה 6
flipped = cv2.flip(image, 1)
cv2.imshow('Flipped Image', flipped)
cv2.waitKey(0)
# תשובה:
# שלב 1: התמונה הופכת על ציר X (מראה אופקית) (flipped).
# שלב 2: התמונה החדשה תוצג עם ההיפוך.

# שאלה 7
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
# תשובה:
# שלב 1: התמונה מומרת לגווני אפור (gray).
# שלב 2: Gradient מחושב בכיוון אופקי (X-axis) באמצעות Sobel (sobelx).
# שלב 3: שינויים בכיוון אופקי יוצגו כתמונה.

# שאלה 8
mask = cv2.inRange(image, (0, 0, 0), (100, 100, 100))
cv2.imshow('Mask', mask)
cv2.waitKey(0)
# תשובה:
# שלב 1: התמונה מומרת למפת מסכה, שבה רק פיקסלים בטווח הצבעים (0,0,0) עד (100,100,100) נשארים, וכל השאר הופכים לשחור (mask).
# שלב 2: התוצאה תוצג כתמונה בינארית עם פיקסלים לבנים ושחורים בלבד.

# שאלה 9
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(gray)
cv2.imshow('Equalized', equ)
cv2.waitKey(0)
# תשובה:
# שלב 1: התמונה מומרת לגווני אפור (gray).
# שלב 2: ניגודיות התמונה מותאמת באופן אחיד באמצעות equalization (equ).
# שלב 3: התמונה תוצג עם שיפור בניגודיות.

# שאלה 10
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(gray, kernel, iterations=1)
cv2.imshow('Dilated Image', dilated)
cv2.waitKey(0)
# תשובה:
# שלב 1: התמונה מומרת לגווני אפור (gray).
# שלב 2: התמונה מורחבת בעזרת מבנה של כותרת (kernel) בגודל 3x3.
# שלב 3: התוצאה תוצג עם אזורים שהורחבו כדי להדגיש תכנים בתמונה.

# שאלה 11
video_capture = cv2.VideoCapture('test1.mp4')

# קריאת הפריים הראשון
success, frame = video_capture.read()

if success:
    # שמירת הפריים כתמונה
    cv2.imwrite('video_frame.jpg', frame)
    print("Frame saved as 'video_frame.jpg'")
else:
    print("Failed to read the frame from the video.")

# שחרור משאבים
video_capture.release()
# סיום
cv2.destroyAllWindows()
