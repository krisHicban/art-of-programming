import random

# --------------------------
# Helper functions
# --------------------------

def create_deck():
    """Return a list of cards in a deck."""
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    deck = [(value, suit) for suit in suits for value in values]
    random.shuffle(deck)
    return deck

def card_value(card):
    """Return the Blackjack value of a single card."""
    value, _ = card
    if value in ['J', 'Q', 'K']:
        return 10
    elif value == 'A':
        return 11  # We'll handle Ace flexibility in hand_value
    else:
        return int(value)

def hand_value(hand):
    """Calculate the value of a hand, considering Aces as 1 or 11."""
    value = sum(card_value(card) for card in hand)
    # Adjust for Aces
    aces = sum(1 for card in hand if card[0] == 'A')
    while value > 21 and aces:
        value -= 10
        aces -= 1
    return value

def deal_card(deck):
    """Pop a card from the deck."""
    return deck.pop()

def deal_initial_hands(deck):
    """Deal initial hands to player and house."""
    player_hand = [deal_card(deck), deal_card(deck)]
    house_hand = [deal_card(deck), deal_card(deck)]
    return player_hand, house_hand

def display_hand(hand, hidden=False):
    """Display cards. If hidden, hide the first card."""
    if hidden:
        print("[Hidden]", hand[1])
    else:
        print(", ".join([f"{v} of {s}" for v, s in hand]))

# --------------------------
# Game logic
# --------------------------

def player_turn(deck, player_hand):
    while True:
        print("\nYour hand:")
        display_hand(player_hand)
        print("Value:", hand_value(player_hand))
        
        if hand_value(player_hand) > 21:
            print("Bust! You exceeded 21.")
            return player_hand, True  # True = bust
        
        action = input("Do you want to Hit or Hold? (h/H for Hit, s/S for Hold): ").lower()
        if action == 'h':
            player_hand.append(deal_card(deck))
        elif action == 's':
            break
        else:
            print("Invalid input, please choose again.")
    return player_hand, False

def house_turn(deck, house_hand, player_value):
    print("\nHouse's turn:")
    display_hand(house_hand)
    while hand_value(house_hand) < player_value and hand_value(house_hand) <= 21:
        print("House hits!")
        house_hand.append(deal_card(deck))
        display_hand(house_hand)
    
    value = hand_value(house_hand)
    if value > 21:
        print("House busts!")
    else:
        print("House stands with value:", value)
    return house_hand

def determine_winner(player_hand, house_hand, player, bet=100):
    player_value = hand_value(player_hand)
    house_value = hand_value(house_hand)
    
    if player_value > 21:
        print("You lose this round.")
        player['budget'] -= bet
    elif house_value > 21 or player_value > house_value:
        print("You win this round!")
        player['budget'] += bet
    elif player_value < house_value:
        print("You lose this round.")
        player['budget'] -= bet
    else:
        print("It's a tie!")
    print("Your current budget:", player['budget'])

# --------------------------
# Main functions
# --------------------------

def seat_player(name, budget):
    """Create a player dictionary."""
    return {'name': name, 'budget': budget}

def run_blackjack(player):
    deck = create_deck()
    player_hand, house_hand = deal_initial_hands(deck)
    
    # Player's turn
    player_hand, busted = player_turn(deck, player_hand)
    
    if not busted:
        # House turn
        house_hand = house_turn(deck, house_hand, hand_value(player_hand))
    
    # Reveal results
    print("\nFinal hands:")
    print("Player:")
    display_hand(player_hand)
    print("House:")
    display_hand(house_hand)
    
    # Determine winner
    determine_winner(player_hand, house_hand, player)

# --------------------------
# Example run
# --------------------------

if __name__ == "__main__":
    player = seat_player("Alice", 1000)
    run_blackjack(player)
