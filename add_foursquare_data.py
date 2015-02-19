import foursquare_checkin as fs
from users import *
import tweepy
import numpy as np

"""jb = pickle.load(open('data/hiyaholls.pkl', 'rb'))

venues_list = []
check_in_dates = []

for check_in in jb.foursquare.check_ins:
	try:
		_, ven_id = fs.get_venue_from_checkin(check_in.entities['urls'][0]['expanded_url'])
		#print check_in.entities['urls'][0]['expanded_url']
		try:
			check_in_dates.append(check_in.created_at)
			venue = fs.get_venue(ven_id) 
			#print venue
			venues_list.append([check_in, venue])
		except:
			# venue id has a problem
			venues_list.append([check_in, 'null'])
			pass
	except:
		# venue doesn't exist
		venues_list.append([check_in, 'null'])
		pass

jb.foursquare.check_ins = venues_list

print jb.foursquare.check_ins[0][0].text.encode('cp850', errors='replace').decode('cp850')
print jb.foursquare.check_ins[0][1]['venue']['name']
"""
def fill_foursquare_data(pickle_name):
	jb = pickle.load(open('data/'+ pickle_name + '.pkl', 'rb'))
	venues_list = []
	check_in_dates = []

	for check_in in jb.foursquare.check_ins:
		try:
			_, ven_id = fs.get_venue_from_checkin(check_in.entities['urls'][0]['expanded_url'])
			print check_in.id
			try:
				check_in_dates.append(check_in.created_at)
				venue = fs.get_venue(ven_id) 
				#print venue
				venues_list.append([check_in, venue])
			except:
				# venue id has a problem
				venues_list.append([check_in, 'Fetching Error'])
				pass
		except:
			# venue doesn't exist
			venues_list.append([check_in, '404 Error'])
			pass

	print len(venues_list)
	jb.foursquare.check_ins = venues_list
	pickle.dump(jb,open('data/post/'+pickle_name+'.pkl','wb'))	

def process_files(filepath):
	filenames = os.listdir(filepath)
	existing_files = os.listdir(filepath+'post/')
	for filename in filenames:
		print filename
		if filename in existing_files:
			"file exists"
			continue

		info = os.stat(filepath+filename)
		if info.st_size > 4000 and filename!='post':
	 		nameoffile = re.split('\.',filename)[0]
			print nameoffile
			fill_foursquare_data(nameoffile)

if __name__ == '__main__':
	process_files('C:/Users/Maria/Desktop/thesis/data/')