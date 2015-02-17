import foursquare
import cPickle as pickle
#import plots

#credentials
import foursqid

# Connect to Foursquare API 
client = foursquare.Foursquare(client_id=foursqid.client_id, client_secret=foursqid.client_secret)
auth_uri = client.oauth.auth_url()

# Query for categories
categories_raw = client.venues.categories()

def map_lowest_to_parent(categories_raw):
	""" Map categories to toppest category. E.g: Museum -> A&E """
	mapping_lowest_to_maincategory = {}

	for top_category in categories_raw['categories']:
			mapping_lowest_to_maincategory[top_category['id']] = top_category['id']
			for sub_category in top_category['categories']:
				mapping_lowest_to_maincategory[sub_category['id']] = top_category['id']
				for sub_sub_category in sub_category['categories']:
					mapping_lowest_to_maincategory[sub_sub_category['id']] = top_category['id']

	return mapping_lowest_to_maincategory

def map_lowest_to_sub_parent(categories_raw):
	""" Map categories to toppest category. E.g: Art Museum -> A&E """

	mapping_lowest_to_subcategory = {}

	for top_category in categories_raw['categories']:
			mapping_lowest_to_subcategory[top_category['id']] = top_category['id']
			for sub_category in top_category['categories']:
				mapping_lowest_to_subcategory[sub_category['id']] = sub_category['id']
				for sub_sub_category in sub_category['categories']:
					mapping_lowest_to_subcategory[sub_sub_category['id']] = sub_category['id']

	return mapping_lowest_to_subcategory

def map_id_to_name(categories_raw):
	""" Just for readability """
	map_id_to_name = {}

	for top_category in categories_raw['categories']:
		map_id_to_name[top_category['id']] = top_category['name']

		for sub_category in top_category['categories']:
			map_id_to_name[sub_category['id']] = sub_category['name']

			for sub_sub_category in sub_category['categories']:
				map_id_to_name[sub_sub_category['id']] = sub_sub_category['name']

	return map_id_to_name

if __name__=='__main__':
	mapping = map_id_to_name(categories_raw)
	mapping_lowest_to_subcategory = map_lowest_to_sub_parent(categories_raw)
	mapping_lowest_to_maincategory = map_lowest_to_parent(categories_raw)
	pickle.dump(mapping,open('mappingnameid.pkl','wb'))
	pickle.dump(mapping_lowest_to_maincategory,open('mapmain.pkl','wb'))
	pickle.dump(mapping_lowest_to_subcategory,open('mapsub.pkl','wb'))