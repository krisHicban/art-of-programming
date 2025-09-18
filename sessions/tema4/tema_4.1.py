# Smart Playlist

songs = [
    {"title": "Blinding Lights", "artist": "The Weeknd", "genre": "pop", "duration": 200, "rating": 4.8},
    {"title": "Shape of You", "artist": "Ed Sheeran", "genre": "pop", "duration": 230, "rating": 4.5},
    {"title": "Nose Bleeds", "artist": "Doechii", "genre": "HipHop", "duration": 210, "rating": 4.7},
    {"title": "Estudio 5", "artist": "Bach", "genre": "classical", "duration": 360, "rating": 4.9},
    {"title": "Lo-Fi Beats", "artist": "Various", "genre": "lofi", "duration": 180, "rating": 4.4},
    {"title": "Wait and Bleed", "artist": "Slipknot", "genre": "rock", "duration": 290, "rating": 4.6},
]

# Show songs
def display_songs(song_list):
    if not song_list:
        print("\nNo songs found!\n")
        return
    for song in song_list:
        print(f"{song['title']} by {song['artist']} | {song['genre']} | "
              f"{song['duration']}s | Rating: {song['rating']}")
    print()

# Mood-based recommendation
def mood_recommendation(mood, song_list):
    mood_map = {
        "party": ["pop", "electronic", "rock"],
        "study": ["lofi", "classical"],
        "relax": ["lofi", "classical", "pop"]
    }
    genres = mood_map.get(mood.lower(), [])
    return [song for song in song_list if song["genre"] in genres]

# Main program
while True:
    print("\n Smart Playlist Manager ")
    print("1. Show all songs")
    print("2. Filter by artist")
    print("3. Filter by genre")
    print("4. Filter by maximum duration")
    print("5. Filter by minimum rating")
    print("6. Mood-based recommendation")
    print("0. Exit")

    choice = input("\nChoose an option: ")

# Show all songs
    if choice == "1":
        display_songs(songs)

# Filter by artist
    elif choice == "2":
        artist = input("Enter artist name: ")
        filtered = [s for s in songs if artist.lower() in s["artist"].lower()]
        display_songs(filtered)

# Filter by genre
    elif choice == "3":
        genre = input("Enter genre: ")
        filtered = [s for s in songs if s["genre"].lower() == genre.lower()]
        display_songs(filtered)

# Filter by maximum duration (no try)
    elif choice == "4":
        max_dur = int(input("Enter max duration (seconds): "))
        filtered = [s for s in songs if s["duration"] <= max_dur]
        display_songs(filtered)

# Filter by minimum rating (no try)
    elif choice == "5":
        min_rate = float(input("Enter minimum rating: "))
        filtered = [s for s in songs if s["rating"] >= min_rate]
        display_songs(filtered)

# Mood-based recommendation
    elif choice == "6":
        mood = input("Enter mood (party/study/relax): ")
        display_songs(mood_recommendation(mood, songs))

  
    elif choice == "0":
        print("\nGoodbye!")
        break

    else:
        print("\nInvalid choice.")
