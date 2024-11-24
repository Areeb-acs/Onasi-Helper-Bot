import os
import pyodbc

def fetch_query_results():
    """
    Function to connect to SQL Server, execute a query, and return the results.
    
    Returns:
        list: List of rows containing the query results.
    """
    # Define connection parameters
    server = 'localhost\\SQLEXPRESS'  # Replace with your server name/IP
    database = 'AreebBlogDB'  # Replace with your database name

    # Create the connection string
    connection_string = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;"
    )

    try:
        # Establish the connection
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        print("Connection established successfully.")

        # Define your query
        query = """
            SELECT TOP (1000) [TypeId]
                ,[TypeCategory]
                ,[TypeName]
                ,[Description]
                ,[CodeValue]
                ,[CodeDisplayValue]
                ,[CodeDefinition]
                ,[LongDescription]
            FROM [AreebBlogDB].[dbo].[RCM_dataset]
        """

        # Execute the query
        cursor.execute(query)

        # Fetch all results
        rows = cursor.fetchall()

        # Close the cursor and connection
        cursor.close()
        conn.close()

        # Return the query results
        return rows

    except pyodbc.Error as e:
        print("Error in connection or query execution:", e)
        return None  # Return None if there was an error

# Example usage of the function
if __name__ == "__main__":
    results = fetch_query_results()
    if results:
        print("Query Results:")
        for row in results:
            print(row)
    else:
        print("No results or error occurred.")
