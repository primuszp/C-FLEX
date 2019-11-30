#!/bin/bash
# For Windows, use Microsoft Access to convert.
# For Unix, use this bash script to automate the process.

# Convert Microsoft Access Database to MySQL.
# Usage: sh mdb2sql.sh [.mdb] [database name]
#   $1: .mdb file name
#   $2: database name in MySQL you want to create
# Example: sh mdb2sql.sh Vehicles.mdb erdc
# The program will prompt 3 times for SQL password.
# Note:
#   brew install mdbtools

# create database in MySQL
mysql -u root -e "drop database if exists $2; create database $2;"

# get schema (i.e. data structures) of tables and pipe to MySQL
mdb-schema $1 mysql | mysql -u root $2
# save schema
# mdb-schema $1 mysql > schema.sql

# convert all tables in .mdb into intermediate .sql file
SQL="$2.sql"
[ -f $SQL ] && rm $SQL

TABLES=$(mdb-tables -1 $1)
for t in $TABLES
do
    if [ "$t" != "Setup" ] && [ "$t" != "Versions" ] && [ "$t" != "VersionChanges" ] && [ "$t" != "Pictures" ]; then # Setup table has a "LastDate" format, which is hard to deal with
        printf "Converting Table: $t..."
        # Option 1: wrap with '"()"' (not good, inconsistent with Windows)
        # mdb-export -I mysql $1 $t | \
        # sed -e 's/)$/)\;/g;s/'\''/./g;s/\("[^"]*"\)/'\''\1'\''/g' >> $SQL
        # Goal: ("WES*8YY)U);F*6&YT`", 0, "Alison", ...) --> ('"WES*8YY)U);F*6&YT`"', 0, '"Alison"', ...), i.e. find all occurence of "xxx" and wrap it as '"xxx"'
        # s/xxx/yyy/g: replace xxx with yyy, 'g' for find all
        # s/)$/)\;/g: to separate lines
        # s/\("[^"]*"\)/'\''\1'\''/g: regex, \(xxx\) for back reference xxx later as \1, ^" for not ", * for Kleene

        # Option 2: wrap with '()' (very subtle issue with 'WES*B(N1@-*I6SZ`8\', the last \ will escape '! we should change every \ to \\...)
        # mdb-export -I mysql -q "'" $1 $t >> $SQL # -q specify how to wrap the text with single/double quotes

        # Option 3: final solution
        mdb-export -I mysql -q "'" $1 $t | \
        sed -e 's/\\/\\\\/g' >> $SQL
        echo "Done"
    fi
done

# import to MySQL database
printf "Importing to MySQL database: $2..."
mysql -u root $2 < $SQL
echo "Done"

rm $SQL

echo "Conversion Success!"
