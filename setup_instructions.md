# Database initialization

## Required Resources

Full execution this full system on a daily basis requires installation of the proper python requirements, a mongo database, and memberships to the following data services.  

Your config.py file should reflect your python environment, your mongodb location and credentials, and any other credentials for required data services. 

### MongoDB

All code assumes you have a local database set up and accessible at the URI "mongodb://localhost:27017/" (Replace with the cloud uri if using a cloud database) with a database created within the client called NBA.  These can be changed within your config.py file.   

### Data Services


# Backfill

Navigate to automation/ and update permissions on Backfill.sh (chmod 755 Backfill.sh)

Simply running ./Backfill.sh will initialize your mongo database if your database and credentials have been properly setup. 


