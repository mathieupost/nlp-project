import csv
import sys
import time
from os import system

user_name = "[[YOUR NAME]]"
filename = "../output/url-versions-2015-06-14-clean-test-errors.csv"

with open(filename) as datafile:
    length = len(datafile.readlines()) - 1

with open(filename) as datafile:
    data = csv.reader(datafile)
    labels = {"t" : "for", "o": "observing", "f": "against"}

    score = 0
    run = -1

    data.next()
    for row in data:
        run += 1
        print "Score: " + str(score) + '/' + str(length)
        print "Progress: " + str(run) + '/' + str(length)
        print "Claim: " + row[1]
        print "Article headline: " + row[2]
        print "Does this article..."
        print "[t] State that the claim is true "
        print "[f] State that the claim is false"
        print "[o] Merely observe the claim, without assessment of its veracity"

        label = raw_input()
        while label not in ['t', 'f', 'o']:
            print "Wrong input, try again..."
            label = raw_input()

        predicted_label = labels[label]

        system('clear')

        if predicted_label == row[3]:
            with open(user_name + "_correct.csv", 'a') as correct_file:
                writer = csv.writer(correct_file)
                writer.writerow(row)
                score += 1

        else:
            with open(user_name + "_incorrect.csv", 'a') as incorrect_file:
                row[4] = predicted_label
                writer = csv.writer(incorrect_file)
                writer.writerow(row)

    print "Final accuracy: " + str(float(score)/float(length))