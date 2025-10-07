import random

# Setup functions

def seat_player(name, budget):
    return {"name": name, "budget": budget, "hand": []}

def build_deck():
    suits = ['♠','♥','♦','♣']
    ranks = ['A',2,3,4,5,6,7,8,9,10,'J','Q','K']
    deck = [(r,s) for r in ranks for s in suits]
    random.shuffle(deck)
    return deck

def deal_card(deck):
    return deck.pop()

def hand_value(hand):
    value = 0
    aces = 0
    for r, s in hand:
        if r in ['J','Q','K']:
            value += 10
        elif r == 'A':
            aces += 1
            value += 11
        else:
            value += r
    while value > 21 and aces:
        value -= 10
        aces -= 1
    return value

#  Game mechanics

def player_turn(player, deck):
    #Handle player's decisions: Hit or Hold
    while True:
        value = hand_value(player["hand"])
        print(f"Your hand: {player['hand']} (value {value})")
        if value > 21:
            print("Bust! You lose this round.")
            return False
        action = input("Hit or Hold? (h=Hit, s=Stand): ").lower()
        if action == 'h':
            player["hand"].append(deal_card(deck))
        else:
            return True

def house_turn(house, player_value, deck):
    #House draws until it beats player or busts
    while hand_value(house["hand"]) < player_value and hand_value(house["hand"]) <= 21:
        house["hand"].append(deal_card(deck))

def decide_winner(player, house, bet=100):
    player_value = hand_value(player["hand"])
    house_value = hand_value(house["hand"])
    print(f"House hand: {house['hand']} (value {house_value})")

    if player_value > 21:
        print("You busted. House wins.")
        player["budget"] -= bet
    elif house_value > 21 or player_value > house_value:
        print("You win!")
        player["budget"] += bet
    elif player_value < house_value:
        print("House wins.")
        player["budget"] -= bet
    else:
        print("It's a tie!")
    print(f"Your new budget: {player['budget']}")

# --- Main game ---

def run_blackjack(player):
    deck = build_deck()
    house = {"hand": []}
    bet = 100

    # initial deal
    player["hand"] = [deal_card(deck), deal_card(deck)]
    house["hand"] = [deal_card(deck), deal_card(deck)]

    print(f"House shows: {house['hand'][0]} and a hidden card")

    # player turn
    if not player_turn(player, deck):
        decide_winner(player, house, bet)
        return

    # house turn
    player_value = hand_value(player["hand"])
    house_turn(house, player_value, deck)

    # final result
    decide_winner(player, house, bet)
    return
# --- Game start ---

name = input("Enter your name: ")
budget = int(input("Enter your budget: "))
player = seat_player(name, budget)
run_blackjack(player)
if player["budget"] <= 0:
    print("You're out of money! Game over.")
