# QnA Flows Affected By Missing Inventory Data

## Purpose

This document shows example customer conversations where the chatbot collects useful information, but the available trailer data is missing important values such as:

- length
- width
- height
- payload capacity
- hitch type
- other fields that can contribute to finding a trailer such a gate type, wheels,rear door, suspension etc

Because those values are missing in the inventory data, the chatbot may fetch results that are technically related but cannot be fully verified against the customer's exact request.


---

## Flow 1: Aluma Utility Trailer With Payload Need

### Why This Flow Can Produce Weak Results

The spreadsheet has `UTILITY / Aluma` listings where the important fit fields are missing. In the reviewed rows, length, width, height, payload capacity, and hitch type were not filled in.

So if the customer asks for an Aluma utility trailer and gives a weight need, the chatbot can understand the request, but the inventory records may not prove whether the shown Aluma trailers can actually handle the load.

### Example Conversation

1. Customer: `I need an Aluma utility trailer.`
2. Bot: `What will you be hauling on the utility trailer?`
3. Customer: `heavy equipment.`
4. Bot: `What's the rough total weight of your load?`
5. Customer: `Around 1800 lbs.`
6. Bot: `Do you have a size preference (length / width)?`
7. Customer: `No preference.`

### What The Chatbot Understands

- trailer type: `Utility`
- preferred make: `Aluma`
- use case: hauling heavy equipment
- rough load weight: `1800 lbs`
- size preference: none

### What Can Go Wrong Because Of Missing Data

The chatbot may fetch Aluma utility listings because the category and make match. However, many of those records do not contain the payload capacity, length, width, height, or hitch type.

That means the chatbot cannot confidently confirm:

- whether the trailer can handle `1800 lbs`
- what size the trailer is
- what hitch type it has
- whether it is truly the best fit compared with other utility options

### How Ranking Helps

After the initial results are fetched, the ranking step tries to put better-fitting trailers higher. However, if the Aluma utility records are missing the key fit values, the ranking step has limited information to work with.

In this case, the chatbot may still show related Aluma utility trailers, but the fit is less certain because the source data is incomplete.

---

## Flow 2: Diamond C Equipment Trailer For A Heavy Skid Steer

### Why This Flow Can Produce Weak Results

The spreadsheet has `Equipment / Diamond C` listings where length and hitch are often present, but width, height, and payload capacity are missing.

That matters because equipment trailer customers often care about load weight and deck width.

### Example Conversation

1. Customer: `I'm looking for a Diamond C equipment trailer.`
2. Bot: `What equipment will you be hauling?`
3. Customer: `A skid steer.`
4. Bot: `What's the rough total weight of the load?`
5. Customer: `About 10,000 lbs.`
6. Bot: `About how long is the load, or what deck length do you need?`
7. Customer: `24 ft.`
8. Bot: `Do you prefer a bumper pull or gooseneck hitch?`
9. Customer: `Gooseneck.`

### What The Chatbot Understands

- trailer type: `Equipment`
- preferred make: `Diamond C`
- haul item: skid steer
- rough load weight: `10,000 lbs`
- requested length: `24 ft`
- hitch preference: `Gooseneck`

### What Can Go Wrong Because Of Missing Data

The chatbot can search for Diamond C equipment trailers with the right category, make, length, and hitch. But several Diamond C equipment records are missing payload capacity and width.

That means the chatbot may find a related 24 ft Diamond C equipment trailer, but it may not be able to confirm:

- whether the trailer can safely handle `10,000 lbs`
- whether the width is suitable for the skid steer
- whether it is truly the best equipment trailer for that load

### How Ranking Helps

The ranking step can prefer listings that have better matching length, payload, width, and height information. But if the Diamond C equipment records are missing payload and width, the ranking step cannot fully verify the most important fit details.

So the final results may be relevant by category, make, length, and hitch, but still weaker on load-fit confidence.

---

## Flow 3: Texas Pride Dump Trailer With Material And Capacity Need

### Why This Flow Can Produce Weak Results

The spreadsheet has `Dump / Texas Pride` listings where length and hitch are present, but width, height, and payload capacity are missing.

That matters because dump trailer customers usually care about:

- material being hauled
- load weight
- bed size
- side height

### Example Conversation

1. Customer: `I need a Texas Pride dump trailer.`
2. Bot: `What material will you be hauling?`
3. Customer: `Construction debris.`
4. Bot: `What's the rough haul weight per load?`
5. Customer: `Around 12,000 lbs.`
6. Bot: `Do you have a preference for the dump mechanism, scissor lift, telescopic, or standard?`
7. Customer: `telescopic.`

### What The Chatbot Understands

- trailer type: `Dump`
- preferred make: `Texas Pride`
- haul material: construction debris
- rough load weight: `12,000 lbs`
- dump mechanism preference: telescopic

### What Can Go Wrong Because Of Missing Data

The chatbot may fetch Texas Pride dump trailers because the category and make match. However, the reviewed Texas Pride dump records do not include payload capacity, width, or height or a dumping mechanism type.

That means the chatbot cannot confidently confirm:

- whether the trailer can handle `12,000 lbs`
- whether the bed is wide enough
- whether the side height is suitable for construction debris
- whether the dumping mechanism is telescopic
- whether the displayed trailer is the best fit for that material and weight

### How Ranking Helps

The ranking step can still compare available listings and prefer stronger matches when the data exists. But if the Texas Pride dump records are missing payload, width, and height, the ranking step has less information to work with.

So the chatbot may return relevant dump trailer options, but the customer-facing fit should be treated as less certain unless the missing details are confirmed by the team.

---

## Main Takeaway

In these flows, the chatbot is not misunderstanding the customer. The issue is that the inventory data is missing important trailer-fit values.

When the data is complete, the chatbot can search and rank results more confidently. When the data is incomplete, the chatbot may still show related trailers, but it cannot fully prove that every result matches the customer's exact needs.


