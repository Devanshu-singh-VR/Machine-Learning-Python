#import to excel
import xlwt
from xlwt import Workbook

# Workbook is created
wb = Workbook()

# add_sheet is used to create sheet.
sheet1 = wb.add_sheet('Sheet 1')

for i in range(0,99):
    sheet1.write(i, 0, A[98-i])
    sheet1.write(i, 1, B[98-i])
wb.save('D:\hello\Bexample.xls')