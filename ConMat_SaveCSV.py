import csv
def SaveCSV(filename, ConMat):
    with open(filename, "a", newline='') as f:
        writer = csv.writer(f)
        for r in ConMat:
            writer.writerow(r)