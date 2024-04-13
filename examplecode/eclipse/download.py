import flickrapi

api_key = u'd91fc5f64aa60854f208b4cf181ede3b'
api_secret = u'cc9c8da1bff8d5ee'

flickr = flickrapi.FlickrAPI(api_key, api_secret)

pool_id = '2986814@N24'

# Get a list of photos from the pool
photos = flickr.photosets.getPhotos(photoset_id=pool_id)['photoset']['photo']

# Loop through each photo and print its URL
for photo in photos:
    photo_url = 'https://farm{}.staticflickr.com/{}/{}_{}.jpg'.format(
        photo['farm'], photo['server'], photo['id'], photo['secret'])
    print(photo_url)

    # Optionally, you can download and display the image using PIL
    try:
        image = Image.open(urllib.request.urlopen(photo_url))
        image.show()
    except Exception as e:
        print('Error downloading image:', e)
    break