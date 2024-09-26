""" a module to store data in mongodb and do CRUD instructions

the moduke use api and key to connect to mongodb 
have CRUD protocol for create, read, update and delete

"""

from bson import ObjectId
import pymongo

def connect_mongodb():
    """
      connect to the mongodb database and inform the status
    Args:
    Returns:
      client: the mongodb client 
      to use the client, example:
      db = client['final_test2']
      stock_collection = db['stock price']
    """
    uri = 'mongodb+srv://koyisaa:on5M8suSj0YW3mKy@cluster0.ahdd5cw.mongodb.net/?retryWrites=true&w=majority'
    # Create a new client and connect to the server
    client = pymongo.MongoClient(uri)
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

    return client

def create(database: pymongo.collection.Collection, item) -> bool:
    """
      inset the data in the database collection and return the result
    Args:
      database: the db collection to insert
      item: the json/dict data to be inserted
    Returns:
      result: whether the create is ture
      warning: if the data already exist in
      this colleciton, the result will be false 
    """
    try:
        #check if this collection already have something inside
        if database.count_documents({}) == 0: 
            database.insert_many(item)
            return True
        else:
            print("Item already exists.")
            return False
    except Exception as e:
        print(f"create error occurred: {e}")
        return False
    
def read(database: pymongo.collection.Collection, query: str):
    """
     read the data in the database collection with the query and return the result
    Args:
      database: the db collection to query
      query: the research query,dict/str are ok
    Returns:
      result: the db read query response
    """   
    try:
        results = database.find(query)
        return list(results)
    except Exception as e:
        print(f"find error occurred: {e}")
        return None
    
def update(database: pymongo.collection.Collection, item, properties: dict) -> bool:
    """
      update the data in the database collection and return the result
    Args:
      database: the db collection to update
      item: the json/dict data to be updateed
      properties: the update query parameters
    Returns:
       result: whether the update is success
    """
    try:
        result = database.update_many(item, {"$set": properties})
        return bool(result.modified_count> 0)
    except Exception as e:
        print(f"update error occurred: {e}")
        return False

def delete(database: pymongo.collection.Collection, id: str) -> bool:
    """
     delete the data in the database collection with object id and return the result
    Args:
      database: the db collection to delete item
      id: the object id of the data to be deleted
    Returns:
       result: whether the delete is success
    """
    try:
        obj_id = ObjectId(id) #convert the str to object id type
        result = database.delete_one({"_id": obj_id})
        return bool(result.deleted_count > 0)#count the delete number
    except Exception as e:
        print(f"delete error occurred: {e}")
        return False