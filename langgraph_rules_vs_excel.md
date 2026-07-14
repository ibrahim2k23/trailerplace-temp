# TrailerPlace Chatbot Rules

## Purpose

This document explains how the current TrailerPlace chatbot works

`Trailer_Language_Mapping_agent_ready_v2 4-15-26.xlsx`

It is split into:

- rules that are already reflected in the Excel file and are also being followed in the chatbot
- rules that are being followed in the chatbot but are not clearly written in the Excel file

The document is organized around these four areas:

1. Questions
2. Features and filters set automatically
3. Tool calls and when they are used
4. Special-purpose helper agents

---

## Section 1: What The Excel Sheet Covers And The Chatbot Also Follows

### 1. Questions

The chatbot follows the same overall logic as the Excel file:

- first understand what kind of trailer the customer is asking about
- then ask the most important questions for that trailer type
- then ask helpful follow-up questions if needed
- once enough information is collected, show recommendations

Examples of category-based question flow:

- `Equipment`
  - asks what the customer is hauling
  - asks the rough weight
  - asks the rough length
  - asks bumper pull or gooseneck
- `Utility`
  - asks what the customer is hauling
  - asks the rough weight
  - may ask about size preference
- `Dump`
  - asks what material will be hauled
  - asks the rough weight
  - may ask about dump style preference
- `Enclosed`
  - asks what the trailer will be used for
  - asks about the needed size
  - may ask about inside features
- `Livestock`
  - asks what trailer length is needed
  - may ask about gate preference

The chatbot also follows the workbook idea of asking the next best question in order, rather than jumping around randomly.

If the customer does not answer a required question clearly, the chatbot tries again in a natural way instead of starting the whole conversation over.

### 2. Features And Filters Set Automatically

The chatbot follows the workbook idea that customer answers should be turned into useful search details.

For example:

- if the customer gives a trailer length, that becomes a search detail
- if the customer gives a width, that becomes a search detail
- if the customer gives a rough load weight, that becomes a search detail
- if the customer gives a hitch preference, that becomes a search detail
- if the customer gives a budget, that becomes a search detail

The chatbot also uses category mapping values so different customer wording can lead to the correct trailer category.

Here is the category mapping in a simpler client-friendly format:

- `Aluminum`
  - common wording: aluminum, lightweight, won't rust
- `Car Hauler`
  - common wording: car hauler, toy hauler, trailer without sides
- `Diesel Tank`
  - common wording: diesel tank, fuel tank, tank trailer
- `Dump`
  - common wording: dump trailer, scissor lift dump, hoist dump, telescopic dump, front lift dump
- `Enclosed`
  - common wording: enclosed trailer, box trailer, cargo trailer, v-nose
- `Equipment`
  - common wording: equipment trailer, lowboy, low profile, skid steer trailer, mini excavator trailer, tractor trailer, deckover
- `Fiber`
  - common wording: fiber trailer, splicing trailer, fiber optic trailer
- `Flatbed`
  - common wording: flatbed, hotshot, step deck, dovetail, platform trailer
- `Livestock`
  - common wording: livestock trailer, cattle trailer, goat trailer, hog trailer, Galyean, Star trailer, Calico trailer
- `Race Trailer`
  - common wording: race trailer, enclosed car hauler
- `Roll Off`
  - common wording: roll off, roll-off, dumpster trailer
- `Tilt`
  - common wording: tilt trailer, full tilt, gravity dampened tilt, hydraulic dampened tilt
- `Utility`
  - common wording: utility trailer, landscape trailer, lawnmower trailer, ATV trailer, motorcycle trailer
- `Welding`
  - common wording: welding trailer, welder trailer

Quick examples:

- `cattle` leads to `Livestock`
- `lowboy` leads to `Equipment`
- `hotshot` leads to `Flatbed`
- `landscape trailer` leads to `Utility`
- `roll-off` leads to `Roll Off`

There is also a special clarification rule for office-style trailers:

- if the customer says `office trailer` or `cooldown trailer`
- the chatbot checks whether they mean a fiber / telecom use case
- if yes, it treats it as `Fiber`
- if not, it treats it as `Enclosed`
- if it is still unclear, it asks a clarification question

### 3. Tool Calls And When They Are Used

The chatbot follows the workbook's overall intent-routing and handoff idea, but with actual tool actions behind it.

The main tool actions are:

- `Search for trailers`
  - used after enough information has been collected
  - returns recommended trailer results
- `Interest email`
  - used when the customer is interested in a specific shown trailer
- `FAQ / support email`
  - used for contact questions, financing, trade-in, service, parts, or store information
- `Escalation alert email`
  - used when the customer asks the business to do something the chatbot cannot do directly

### 4. Special-Purpose Helper Agents

The Excel file is structured like a guided sales process, and the chatbot follows the same idea through focused helper decision-makers.

In simple terms, the chatbot has separate helpers for:

- understanding what the customer is asking for
- understanding which trailer category fits
- deciding what question to ask next
- deciding whether the customer needs a recommendation, help information, or a handoff
- reviewing search results before showing them

So while the workbook shows the business logic, the chatbot applies that logic step by step during the conversation.

---

## Section 2: Rules The Chatbot Follows Beyond The Excel Sheet

### 1. Questions

The chatbot has some extra behavior that goes beyond the workbook.

#### Generic early-stage questions

If the customer has not named a trailer category yet, the chatbot can still save useful information such as:

- what they are hauling
- what they need the trailer for
- size details like length or width

Then, once the customer names the category, the chatbot carries that information into the correct flow.

#### No-preference handling

If the customer says something like:

- `No preference`
- `Either is fine`
- `I am not sure`

the chatbot can leave that detail blank and continue, instead of forcing the customer to choose.

#### Dynamic extra questions when needed

In some situations, the chatbot can add an extra question when it believes that detail is especially important.

For example, it may ask about width for certain heavy-duty situations, even if width was not part of the original basic flow.

### 2. Features And Filters Set Automatically

The chatbot has a few practical defaults and automatic behaviors that are not clearly written in the workbook.

#### Category defaults

Current automatic defaults include:

- `Flatbed`
  - if no width has been provided, the chatbot uses `8 ft` as the default width
- lightweight `Utility` cases
  - in some lighter-duty utility situations, the chatbot may use a light payload assumption to keep the search moving


#### Smart reading of size wording

The chatbot can also understand common size wording such as:

- `6x12`
- `7x16`
- `6x12x5`

and turn that into useful size information.

It can also understand side-height wording such as:

- `3 ft sides`
- `3 inch walls`

and treat that as a height-related detail.

#### Brand-based reranking inside each category

After the chatbot pulls possible trailers, it can give extra preference to brands that make more sense for that trailer category.

In simple terms:

- the chatbot does not treat every brand the same across every category
- it can give stronger priority to brands that are more relevant for the category the customer is shopping in
- this helps the final recommendations feel more appropriate even when multiple trailers match the basic search details

### 3. Tool Calls And When They Are Used

The chatbot has several practical tool-call rules that are not clearly laid out in the workbook.

#### Contact-first sending

If the chatbot needs to send an email action but does not yet have contact details, it will:

- ask for phone number or email
- wait for the customer to provide it
- then send the action

This applies to:

- interest email
- FAQ / support email
- escalation alert email

#### Recommendation search versus direct item follow-up

The chatbot uses one flow to recommend trailers and a different follow-up behavior after results have already been shown.

That means:

- before results are shown, it focuses on qualification and recommendation
- after results are shown, it can answer follow-up questions about the shown trailers more directly

#### Catalogue-style requests

If the customer is asking generally what TrailerPlace carries, the chatbot can treat that differently from a detailed recommendation request.

#### Confusion or repeated frustration

If the customer seems stuck, confused, or frustrated repeatedly, the chatbot can move toward a handoff path rather than just continuing to ask more questions.

### 4. Special-Purpose Helper Agents

The current chatbot has more internal helper behavior than the workbook describes.

In plain language, there are helpers for:

- understanding contact details
- understanding whether the customer is asking for general browsing or a real recommendation
- understanding when a business-action request needs escalation
- understanding whether the customer answered the current question
- improving the order of results after the first search

These are implementation details, but the main point for the client is:

- the chatbot is doing more behind the scenes than the workbook alone shows

---

### 5. Default Values And Automatic Defaults

This section lists the current automatic defaults in a simple way.

#### Current Automatic Defaults

- `Flatbed`
  - if width is not provided, the chatbot uses `8 ft`
- some lighter-duty `Utility` situations
  - the chatbot may use a light payload assumption to keep the search moving
- it can ask an extra width question in selected cases when width seems important:
  `About how wide is the load, or what trailer width do you need?`
- it can leave optional details blank when the customer has no preference

---


### 6. Example Conversation Flow: Utility Trailer

This example shows how the chatbot would handle a normal utility-trailer conversation in business terms.

#### Step 1: Customer names the trailer type

Customer:

`I need a utility trailer.`

What happens:

- the chatbot understands the customer wants a utility trailer
- it starts the utility-trailer question flow

Question asked:

`What will you be hauling on the utility trailer?`

Possible answers:

- `Anything that the user mentions`


What happens:

- the chatbot saves that as the main hauling use
- then it asks the next important utility-trailer question

Question asked:(this question is only asked if the hauling item isn't classified as light weight)

`What's the rough total weight of your load?`

Possible answers:

- `About 1800 pounds`
- `Around 2,000 lbs`
- `Not sure`

#### Step 3: Customer gives the weight

Customer:

`Around 1800 lbs.`

What happens:

- the chatbot saves that weight information
- then it may ask one more helpful follow-up question

Question asked:

`Do you have a size preference (length / width)?`

Possible answers:

- `6x12`
- `I need 14 ft`
- `No preference`

#### Step 4: Customer has no preference

Customer:

`No preference.`

What happens:

- the chatbot leaves size blank
- it does not force the customer to choose something they do not care about
- because the important questions were already answered, it can still move forward

#### Step 5: The chatbot searches for trailers

At that point, the chatbot has enough to search.

In this example, the key details are:

- trailer type: `Utility`
- hauling use: mower and landscaping tools
- rough weight: `1800 lbs`
- size preference: none

#### How The Search Works In Plain Language

The search uses a small number of details directly, mainly because not every trailer record has every field filled out well enough to rely on it.

The direct search usually leans most on:

- trailer category
- make, if requested
- hitch type, if requested
- length, if clearly given

Then, after the first set of trailers is found, the chatbot improves the order of the results by looking at the rest of the customer's needs more intelligently.

In simple terms:

- first it finds likely matches using the most reliable details
- then it sorts those matches so the better-fitting trailers rise to the top

#### Simple Example Conversation

1. Customer: `I need a utility trailer.`
2. Bot: `What will you be hauling on the utility trailer?`
3. Customer: `A zero turn mower and some landscaping tools.`
4. Bot: `What's the rough total weight of your load?`
5. Customer: `Around 1800 lbs.`
6. Bot: `Do you have a size preference (length / width)?`
7. Customer: `No preference.`
8. Bot:
   - keeps the search focused on utility trailers
   - uses the hauling use and weight to guide the search
   - leaves size blank because the customer has no preference
   - finds likely matches
   - then ranks those matches so the better options appear first

---

### 9. Final Takeaway

The Excel file is still a good business guide for:

- category language
- question flow
- recommendation intent
- handoff intent

The current chatbot follows that foundation, but it also adds extra practical behavior so it can:

- handle unclear answers better
- keep moving when the customer has no preference
- manage contact-based email actions
- handle escalation cases
- search and rank results more effectively with imperfect inventory data
