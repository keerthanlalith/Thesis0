#!/bin/sh
echo "Test 1"
currentDate=`date`
echo $currentDate >Results/data1.txt
python ae1.py >> Results/data1.txt
clear 
echo "Test 2"
currentDate=`date`
echo $currentDate >Results/data2.txt
python ae2.py >> Results/data2.txt
clear 
echo "Test 3"
currentDate=`date`
echo $currentDate >Results/data3.txt
python ae3.py >> Results/data3.txt
clear 
echo "Test 4"
currentDate=`date`
echo $currentDate >Results/data4.txt
python ae4.py >> Results/data4.txt
clear 
echo "Test 5"
currentDate=`date`
echo $currentDate >Results/data5.txt
python ae5.py >> Results/data5.txt
clear 
echo "Test 6"
currentDate=`date`
echo $currentDate >Results/data6.txt
python ae6.py >> Results/data6.txt
clear 
echo "Test 7"
currentDate=`date`
echo $currentDate >Results/data7.txt
python ae7.py >> Results/data7.txt
clear 
echo "Test 8"
currentDate=`date`
echo $currentDate >Results/data8.txt
python ae8.py >> Results/data8.txt
clear 
echo "Test 9"
currentDate=`date`
echo $currentDate >Results/data9.txt
python ae9.py >> Results/data9.txt
echo "Done"

file="Dataall.txt"

if [ -f "$file" ] ; then
    rm "$file"
fi

tail -n 1 Results/data1.txt >> Dataall.txt
tail -n 1 Results/data2.txt >> Dataall.txt
tail -n 1 Results/data3.txt >> Dataall.txt
tail -n 1 Results/data4.txt >> Dataall.txt
tail -n 1 Results/data5.txt >> Dataall.txt
tail -n 1 Results/data6.txt >> Dataall.txt
tail -n 1 Results/data7.txt >> Dataall.txt
tail -n 1 Results/data8.txt >> Dataall.txt
tail -n 1 Results/data9.txt >> Dataall.txt

cp Dataall.txt Dataallbkp.txt

sed 's/\[//g' Dataall.txt > temp.txt
sed 's/\]//g' temp.txt > Dataall.txt
rm temp.txt