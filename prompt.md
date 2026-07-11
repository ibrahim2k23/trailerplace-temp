# Multi-Agent Trailer Chatbot System Design Prompt (Version 1)

I want to design a multi-agent AI chatbot system for a trailer dealership.

## Core Objectives

* Design the system using a multi-agent architecture.
* Focus on scalability, maintainability, and low operational cost.
* The primary use case is assisting customers with trailer selection, answering questions, collecting lead information, and supporting the sales process.

## LLM

* Primary model: GPT-4o Mini.
* GPT-4o Mini should be assumed to be the default model unless explicitly stated otherwise.
* All architectural decisions should consider cost efficiency.

## Output Requirements

Every agent must return **structured outputs** rather than free-form text whenever possible.

* Use consistent schemas.
* Outputs should be machine-readable.
* Responses should be designed so that downstream agents can consume them without additional parsing.
* Avoid requiring another LLM to interpret unstructured text from previous agents.

## Agent Communication

Design every agent as an independent component.

Each agent should:

* Receive structured input.
* Perform one well-defined responsibility.
* Return structured output.
* Minimize unnecessary context passed between agents.
* Share only information required for downstream tasks.

## Reasoning Philosophy

* Do not introduce deterministic rules, hard-coded guards, regex validation, keyword filters, or rule-based routing unless I explicitly request them.
* Assume the LLM is responsible for reasoning and decision-making.
* Keep the design LLM-centric.
* Give structured outputs to help with reasoning

## Optimization Priorities

Prioritize, in order:

1. Low inference cost.
2. Efficient context usage.
3. Modular agent design.
4. Structured communication between agents.
5. Easy extensibility as additional agents are introduced.







## Available Data

The system has access to dealership data that can be used throughout the conversation. The data should be treated as authoritative unless explicitly stated otherwise.

### 1. Trailer Categories

The system has a list of available trailer categories.

Each category includes:

* Category name
* Category description (if available)
* Mapping terms (keywords, synonyms, common names, abbreviations, and phrases that may refer to the category)

The mapping terms are intended to help the LLM understand what trailer category the customer is referring to, even if the customer does not use the exact category name.

Example structure:

```text
Category
├── Name
├── Description
└── Mapping Terms
```

### 2. Category Questions

Each trailer category has its own predefined list of questions that should be asked to qualify the customer's requirements.

These questions are dealership-defined and represent the information required to recommend suitable trailers.

Each category includes:

* Category name
* Ordered list of qualification questions

Example structure:

```text
Category
├── Name
└── Questions
    ├── Question 1
    ├── Question 2
    ├── Question 3
    └── ...
```

The chatbot should treat these questions as available data rather than hardcoded logic. Future categories or questions may be added without requiring changes to the overall system architecture.












## Available Tools

The system has access to the following tools. Agents should invoke these tools when appropriate.

### 1. Escalation Alert Tool

**Purpose**

This tool is used whenever the user requests something that the chatbot is not capable of performing directly.

Examples include (but are not limited to):

* Scheduling a phone call
* Scheduling a meeting
* Requesting a quotation
* Requesting an email to be drafted or sent
* Any other action that requires human intervention

**Behavior**

* The chatbot should recognize that the request requires human assistance.
* The chatbot should invoke the Escalation Alert Tool.
* The tool generates and sends an internal escalation email to the dealership so a human representative can follow up.
* The chatbot should continue the conversation naturally after triggering the escalation unless instructed otherwise.

---

### 2. Email Tools

**Purpose**

This tool is used to send emails regarding the following scenarios:
FAQ scenarios:
- contact_human:
  "You can reach our team at 979-532-1486. Happy to keep helping with your trailer search too!"
- financing:
  "We offer financing. Call 979-532-1486 to speak with our finance team, and I can keep helping narrow down the right trailer."
- trade_in:
  "Our sales team handles trade-in appraisals. Call 979-532-1486."
- service_parts:
  "Our service and parts team can help. Reach them at 979-532-1486."
- store_info:
  "We're located in Wharton, TX. Call 979-532-1486 or visit https://trailerplace.com. We also offer financing and delivery."

Non-FAQ scenarios:
- generic team request:
  "Thanks, I shared that request with the team so they can help you with it."
- escalation:
  "I've passed your query to our team. In the meantime, I can keep helping you narrow down the right trailer."
- listing interest with a selected item:
  "Your interest in the selected trailer has been logged. Our team can follow up. In the meantime, feel free to visit https://trailerplace.com or call 979-532-1486."
- listing interest without a selected item:
  "Your interest has been logged. Our team can follow up. In the meantime, feel free to visit https://trailerplace.com or call 979-532-1486."
- safe listing-interest fallback:
  "Great, I shared your interest in that trailer with the team. They can follow up with you."


**Behavior**

* If the requested information is unavailable to the chatbot, invoke the Non-FAQ Email Tool.
* The tool sends an internal notification so dealership staff can respond to the customer's request.
* The chatbot should politely inform the user that their request has been forwarded for follow-up.


## Contact Information Requirement for Email Tools

Before invoking any email-based tool, the chatbot must check whether sufficient customer contact information is available.

This applies to:

* Escalation Alert Tool
* Non-FAQ Email Tool
* Log Interest Tool, if it requires dealership follow-up

### Minimum Required Contact Information

To send an email-based alert, the chatbot must have:

```text
Name
+
(Email OR Phone Number)
```

### If Contact Information Is Missing

If the user triggers an email-based tool but the chatbot does not have sufficient contact information, the chatbot should ask again for:

* Name
* Email address
* Phone number

The same contact collection rules apply:

* If the user provides name + email, proceed.
* If the user provides name + phone number, proceed.
* If the user provides email or phone number but no name, ask once for their name.
* If the user provides name but no email or phone number, ask once for either email or phone number.
* If the user declines, ignores, or refuses to provide the missing contact information, do not invoke the email tool.

### If Contact Information Is Still Missing

If the user does not provide the required minimum contact information after the follow-up request:

* Do not send the email alert.
* Do not continue asking.
* Politely continue the conversation without triggering the tool.

### Core Rule

Email-based tools should only be invoked when the chatbot has enough information for a human representative to follow up.

---

### 3. Pinecone Search Tool

**Purpose**

This tool retrieves trailer recommendations from the vector database after customer requirements have been collected.

**Invocation Rule**

This is the only deterministic tool invocation currently in the system.

Invoke the Pinecone Search Tool only when one of the following conditions is true:

* Every required qualification question for the selected trailer category has been answered.
* The remaining unanswered questions have been explicitly skipped by the customer, and there are no further questions to ask.

The tool must **not** be invoked before the qualification stage is complete.

Once invoked:

* Pass the collected customer requirements as structured input.
* Retrieve the most relevant trailers from Pinecone.
* Return the results as structured output for downstream processing.
* Do not perform additional deterministic validation before or after the search unless explicitly specified in future requirements.










## User Flow – Contact Information Collection

This section defines how the chatbot should collect customer contact information at the beginning of a conversation.

### Objective

Collect enough information for dealership follow-up while minimizing friction.

### Required Information

The chatbot should attempt to collect:

* Name
* Email address
* Phone number

However, the minimum information required for follow-up is:

* Name
* **AND**
* At least one contact method:

  * Email address, or
  * Phone number

Therefore, the chatbot should aim for:

```
Name
+
(Email OR Phone Number)
```

Having all three fields (Name, Email, and Phone Number) is preferred but not required.

---

### First-Time User

When the user starts a conversation for the first time, the chatbot should politely request:

* Name
* Email address
* Phone number

The user may provide these in any order.

---

### Accepted Outcomes

#### Case 1

User provides:

* Name
* Email
* Phone Number

Result:

* Accept all information.
* Do not ask for contact information again.

---

#### Case 2

User provides:

* Name
* Email

Result:

* Accept.
* Do not request the phone number.

---

#### Case 3

User provides:

* Name
* Phone Number

Result:

* Accept.
* Do not request the email.

---

#### Case 4

User provides:

* Email or Phone Number
* No Name

Result:

* Ask once for the user's name.

---

#### Case 5

User provides:

* Name
* Neither Email nor Phone Number

Result:

* Ask once for either an email address or a phone number.

---

### User Declines Contact Information

Providing contact information is optional.

If, at the beginning of the conversation, the user:

* declines,
* ignores the request,
* changes the subject, or
* begins asking about trailers instead,

the chatbot should:

* immediately continue with the user's request,
* not insist on collecting contact information,
* not interrupt the user's objective.

The conversation should continue naturally.

---

### Partial Declines

If the user provides only part of the requested information and ignores or refuses the remaining fields, the chatbot should not continue asking for them.

Examples:

**Example A**

User provides:

* Name

Chatbot asks for a phone number or email.

User ignores or declines.

Result:

* Continue the conversation.
* Do not ask again.

---

**Example B**

User provides:

* Email or Phone Number

Chatbot asks for the user's name.

User ignores or declines.

Result:

* Continue the conversation.
* Do not ask again.

---

### Collection Rules

* Contact information should only be requested once during the initial interaction.
* The chatbot should never pressure the user to provide personal information.
* If the user declines or ignores any missing contact fields, consider the collection process complete and continue assisting the user.
* The user's willingness to provide contact information should not affect the quality of assistance provided by the chatbot.






















## User Flow – Category Identification and Question Flow

The chatbot acts as a trailer lead specialist. Its objective is to help customers identify the appropriate trailer category, collect their requirements, answer trailer-related questions at any point during the conversation, and ultimately recommend suitable trailers.

### Step 1 – Determine the User's Intent

Every incoming user message should be interpreted based on its intent rather than assuming the user is always answering the current question.

Possible intents include:

1. **General trailer-related question**

   * The user is asking for information about trailers, towing, hauling, payloads, dimensions, use cases, trailer features, or similar topics.
   * Example:

     * "What is a utility trailer used for?"
     * "Can a bumper pull haul a skid steer?"
     * "What's the difference between tandem and single axles?"
   * Behavior:

     * Answer the user's question naturally.
     * If the chatbot is currently in the qualification process, resume the unanswered qualification question afterward (subject to the repetition rules described later).

---

2. **Category exploration**

The user wants help deciding which trailer category fits their needs.

Example:

> "What are my options for hauling heavy equipment?"

Behavior:

* Recommend the relevant trailer categories.
* Briefly explain each.
* Ask which category the user would like to explore.

---

3. **Category selection**

The user explicitly chooses a trailer category.

Example:

> "I'm looking for a utility trailer."

Behavior:

* Map the category.
* Load that category's qualification questions.
* Begin the question flow.

---

4. **Feature-based trailer request**

The user specifies desired trailer features but has not identified the trailer category.

Example:

> "I need a 15-foot trailer with tandem axles and a bumper pull."

Behavior:

* Extract and store every feature already provided.
* Ask which trailer category the user is interested in.
* Do not begin category-specific qualification questions until a category has been identified.

---

5. **Recommendation request with unclear category**

The user wants recommendations but has not identified the trailer category.

Behavior:

* Suggest the most appropriate trailer categories.
* Ask which category the user wants to explore.

---

### Step 2 – Category Mapping

Once the user selects or clearly implies a trailer category:

* Map it using the available category names and mapping terms.
* Load the corresponding qualification questions.
* Begin collecting the remaining information.

The chatbot must distinguish between:

**A category information request**

Example:

> "What is a utility trailer?"

This is an informational question only.

**A category selection**

Example:

> "I want a utility trailer."

This begins the qualification process.

---

### Step 3 – Lead Specialist Behavior

The chatbot should naturally guide conversations toward helping the customer find an appropriate trailer.

After answering informational questions, the chatbot may transition naturally into trailer selection when appropriate.

The chatbot should be helpful and conversational without becoming repetitive or overly pushy.

---

### Step 4 – Pre-fill Known Variables

Each qualification question corresponds to a single variable.

Before asking any question:

* Check whether that variable has already been mentioned anywhere in the conversation.
* If so:

  * Store it.
  * Mark the question as answered.
  * Skip that question.
  * Continue to the next unanswered variable.

Never ask for information that has already been provided.

---

### Step 5 – Qualification Question Flow

After a category has been identified:

* Ask one qualification question at a time.
* Store each answer against its corresponding variable.
* Continue until every question has either been answered or skipped.

---

Step 6 – Handling Interruptions, Counter Questions, and Non-Answers

The qualification flow is interruptible.

At any point while asking qualification questions, the user may:

Ask a general trailer-related question.
Ask for clarification.
Ask a counter question.
Change the topic temporarily.
Respond with information that does not answer the current qualification question.

Examples include:

"What's the difference between tandem and single axles?"
"Can this trailer haul a skid steer?"
"Do you offer financing?"
"How much weight can a bumper pull tow?"
"Why are you asking this?"

When this occurs, the chatbot should:

Pause the qualification flow.
Answer the user's question or address their concern naturally.
Ask the current qualification question one more time.

If the user answers the qualification question after it is repeated:

Store the corresponding variable.
Continue to the next unanswered question.

If the user still does not answer the qualification question after it has been repeated once:

Mark that question as skipped.
Continue to the next unanswered question.

The chatbot should not repeatedly ask the same qualification question beyond this single follow-up attempt.

---

### Step 8 – Explicit Skip

If the user explicitly skips a question:

* Mark that variable as skipped.
* Continue to the next unanswered question.

Examples:

* "Skip this."
* "Next."
* "I don't know."
* "I'd rather not answer."

---

### Step 9 – Skip Remaining Questions

If the user indicates they no longer wish to answer qualification questions and instead wants trailer recommendations:

* Stop asking further qualification questions.
* Mark every remaining unanswered variable as skipped.
* Invoke the Pinecone Search Tool using all collected variables.
* Present the resulting trailer recommendations.

Examples include:

* "Just show me what you have."
* "Show me the trailers."
* "I don't want to answer any more questions."
* "Give me recommendations."

---

### Conversation Principles

* The qualification process is conversational, not linear.
* Users may interrupt with trailer-related questions at any time.
* Every user message should first be interpreted for intent before assuming it answers the current question.
* Always answer legitimate trailer-related questions before resuming qualification.
* Resume the qualification process from exactly where it was paused.
* Never ask for information that has already been collected.
* Respect skipped questions.
* Use the Pinecone Search Tool only after qualification is complete or the user explicitly requests recommendations early.












## Behavior After Pinecone Tool Calls

After trailer recommendations have been shown, the chatbot should continue treating the conversation as active and stateful.

The chatbot must determine the user's intent after results are displayed.

### 1. Search Again / Update Requirements

If the user changes a requirement after seeing results, the chatbot should treat this as a new search request using the updated requirement.

Example:

> User first asks for a 20-foot trailer and receives results.
> Then the user says: “What about 12 feet?”

Behavior:

* Update the stored requirement from 20 feet to 12 feet.
* Keep any other still-relevant collected variables.
* Call the Pinecone Search Tool again.
* Show a new set of matching results.

This also applies when the user changes features such as:

* trailer length
* width
* height
* hitch type
* payload
* brand
---

### 2. Listing Interest

If the user shows interest in one of the displayed listings, the chatbot should invoke the Log Interest Tool.

Examples:

* “I like the second one.”
* “Tell me more about the latest listing.”
* “I’m interested in that Iron Bull.”
* “Can someone contact me about this one?”
* “I want this trailer.”

Behavior:

* Identify which displayed listing the user is referring to.
* Use the stored listing data, especially the listing URL.
* Invoke the Log Interest Tool with the selected listing information and available customer details.
* Continue the conversation naturally after logging the interest.

---

### 3. Store Shown Listings

For every listing shown to the user, store its URL in conversation state.

Purpose:

* Prevent showing the same listing again.
* Support follow-up references like “the first one” or “that latest listing.”
* Allow the Log Interest Tool to identify the correct listing.

The chatbot does not need to implement the internal filtering logic directly in the prompt. The Pinecone fetch and ranking/filtering layer will handle avoiding previously shown or unavailable listings.

---

### 4. Repeated Pinecone Searches and Category Changes

The Pinecone Search Tool may be called multiple times in the same conversation when the user changes requirements, asks for more options, or starts exploring another trailer category.

### Requirement Changes Within the Same Category

If the user changes only one requirement, update only that variable and keep the rest of the collected requirements unchanged.

Example:

> “Instead of 15 feet, make it 20 feet.”

Behavior:

* Update trailer length to 20 feet.
* Keep width, height, payload, hitch type, and other collected variables unchanged.
* Run the Pinecone Search Tool again if the current category question flow is complete.

### Dropping Requirements

If the user asks to remove previous requirements, update the conversation state accordingly.

Examples:

* “Drop the rest of the features.”
* “Forget all that.”
* “Just search by length.”
* “Only use the 20-foot requirement.”

Behavior:

* Remove the requirements the user wants to drop.
* Keep only the requirements the user wants to preserve.
* Continue the question flow or run Pinecone depending on whether the required category questions are complete or skipped.

### Category Change

If the user changes the trailer category at any point, including during the question flow or after Pinecone results, the chatbot should pause and ask which previously collected features should be kept.

Examples:

* “Actually, show me dump trailers.”
* “What about utility trailers instead?”
* “Let’s switch to equipment trailers.”

Behavior:

1. Map the new category.
2. Identify previously collected transferable requirements, such as:

   * length
   * width
   * height
   * payload / weight capacity
   * hitch type
   * axle type
   * intended use
   * budget
   * brand preference
3. Ask the user which of those requirements should be kept for the new category.
4. Keep all, some, or none of the previous requirements based on the user’s answer.
5. Store kept values against the matching question variables for the new category.
6. Ask any remaining unanswered questions for the new category.
7. Call the Pinecone Search Tool once the new category’s question flow is complete or skipped.

If the user says to keep everything, reuse all transferable values.

If the user says to drop everything, clear the previous category-specific requirements and begin the new category flow from scratch.

If the user keeps only some features, preserve only those features and continue asking the remaining questions.

### Core Rule

When the category changes, do not automatically reuse all previous requirements. First ask the user what should stay the same and what should be dropped.


---

### 5. Conversation Principles After Results

* Do not treat showing results as the end of the conversation.
* Continue helping the user refine, compare, or select trailers.
* Detect whether the user wants a new search, more details, comparison, or human follow-up.
* Preserve conversation context across searches.
* Update only the variables the user changes.
* Do not discard useful previous requirements unless the user clearly changes them.







## Global Conversation Behavior

The chatbot should treat the conversation as holistic rather than a fixed sequence of steps.

At any point in the conversation—including before category selection, during the qualification question flow, after Pinecone recommendations, or after multiple recommendation searches—the user may change topics or ask unrelated trailer or dealership questions.

The chatbot should always determine the user's current intent before deciding how to respond.

### Possible User Intents at Any Time

The user may, at any point:

* Ask a general trailer-related question.
* Ask about a specific trailer category.
* Ask about towing, payload, dimensions, axles, hitch types, brakes, or other trailer features.
* Ask about hauling a particular type of equipment or material.
* Ask dealership-related questions.
* Ask a question that requires invoking one of the available tools.
* Change trailer requirements.
* Change trailer category.
* Ask to compare recommendations.
* Ask to search again.
* Express interest in a listing.
* Resume the qualification process.

The chatbot should correctly identify these intents regardless of the current stage of the conversation.

---

### Conversation State

The chatbot should maintain conversation state throughout the session.

Temporary interruptions should **not** reset:

* the selected category,
* collected variables,
* skipped variables,
* previous Pinecone searches,
* previously shown listings,
* or the current qualification question.

After handling an interruption, the chatbot should resume the conversation from the appropriate point unless the user's latest message changes the direction of the conversation.

---

### Tool Invocation

Requests that require one of the available tools may occur at any time.

For example:

* Escalation Alert Tool
* Non-FAQ Email Tool
* Pinecone Search Tool
* Log Interest Tool

The chatbot should determine whether a tool should be invoked based on the user's current intent rather than the current stage of the conversation.

---

### Core Principle

The conversation should feel natural and human rather than linear.

The chatbot should be capable of switching between:

* answering questions,
* collecting requirements,
* refining searches,
* explaining trailer categories,
* invoking tools,
* showing recommendations,
* and continuing previous conversations,

without losing context or forcing the user to restart the conversation.








## Field Extraction and Category Defaults

The chatbot should continuously extract structured information from every user message and store it in conversation state.

This extracted state should help the chatbot skip already-answered questions, prepare Pinecone searches, and preserve user preferences.

### Searchable Filter Fields

The searchable filter fields are:

* Trailer Length
* Trailer Width
* Trailer Height
* Brand Name
* Payload Capacity
* Hitch Type
* Haul Item


rules for extraction: 
if user says something in form of AxB, it means width x length. If user says AxBxC, it means widthxlengthxheight.
if user says 16 by 8, it means 16 feet length and 8 feet width
convert all data into ft for length width height dimensions
convert all weight dimensions(paylaod capacity) to lbs
hitch type is only of either bumper pull or gooseneck

---

### Category Default Values

The system should support plug-and-play category defaults.

Admins may configure default values for a trailer category.

Default values may be configured for:

* Trailer Length
* Trailer Width
* Trailer Height
* Payload Capacity
* Hitch Type

When a category is selected:

* Apply any configured default values for that category.
* Store them in the corresponding variables.
* Treat them as already answered unless the user later changes them.
* Use them to skip matching qualification questions.

Example:

```text
Category: Utility Trailer
Default Width: 83 inches
Default Hitch Type: Bumper Pull
```

If the user selects Utility Trailer, the chatbot should store:

* Width → 83 inches
* Hitch Type → Bumper Pull

The chatbot should not ask those questions unless the user changes or removes those values.

---

### Direct Field Extraction

If the user mentions any searchable filter field, extract and store it.

Example:

> “I want a 12-foot trailer, 6 feet wide, 5 feet tall, with 5,000 lb payload, gooseneck hitch, and Iron Bull brand for hauling a tractor.”

Extract:

* Length → 12 ft
* Width → 6 ft
* Height → 5 ft
* Payload Capacity → 5,000 lb
* Hitch Type → Gooseneck
* Brand Name → Iron Bull
* Haul Item → Tractor

---

### Loose Answers and No-Preference Answers

The chatbot must support loose answers.

Users may answer with phrases such as:

* “Anything can do.”
* “Whatever fits best.”
* “No preference.”
* “Either this or that.”
* “Whatever you recommend.”
* “I’m flexible.”
* “Not sure.”

For numeric fields such as:

* Trailer Length
* Trailer Width
* Trailer Height
* Payload Capacity

If the user gives a clear number, store the number.

If the user gives a numeric range, store the smallest acceptable value from the range.

Example:

> “Something between 15 and 18 feet.”

Store:

* Length → 15 ft

If the user gives a loose answer that cannot be converted into a number, store:

* value → null

This means the user has no fixed preference for that numeric field.

---

### Hitch Type Loose Answers

For hitch type, if the user gives a clear hitch preference, store it.

Examples:

* “Gooseneck”
* “Bumper pull”
* “Either gooseneck or bumper pull”

If the user gives multiple acceptable hitch types, store them as acceptable options.

If the user gives a loose no-preference answer, store:

* Hitch Type → null

This means the user has no fixed hitch preference.

---

### Haul Item Extraction

Haul Item should store whatever the user says they want to haul.

The chatbot should not over-normalize or discard vague hauling descriptions.

Examples:

* “Car”
* “Tractor”
* “Furniture”
* “Wood”
* “Random things”
* “A lot of things”
* “Furniture, wood, and other stuff”
* “I haven’t thought yet, maybe a few things”

Store the user's hauling description as provided, while keeping it usable for downstream search and reasoning.

---

### Brand Name Extraction

If the user mentions a brand preference, store it under Brand Name.

If the user has no brand preference, store:

* Brand Name → null

---

### Non-Metadata Features

If the user mentions any trailer preference that is not one of the searchable filter fields, store it in:

**Non-Metadata Features**

Examples:

* Ramp
* Axle capacity
* Drive-over fenders
* Electric brakes
* Hydraulic jack
* Tarp system
* Spare tire
* Winch
* LED lights
* Toolbox
* Dovetail
* Mesh sides
* Color preference

These preferences should be preserved in conversation state, but they are not directly searchable filters in the current data.

---

### Updating Stored Fields

The extracted conversation state should always reflect the user's latest intent.

If the user changes one field:

> “Actually, make the length 20 feet.”

Update only Trailer Length and keep all other values unchanged.

If the user asks to drop previous requirements:

> “Drop the rest of the features.”
>
> “Forget all that.”
>
> “Just use the length.”

Remove the fields the user wants to drop and preserve only the fields they want to keep.

---

## Category Mapping Priority and Haul Item Lock

### Category Mapping Before a Category Is Chosen

Before a category has been chosen, the chatbot may use the category mapping terms to infer the trailer category.

If the user mentions a mapping term that clearly points to a trailer category, the chatbot may map that category.

Example:

> “I need a trailer to haul livestock.”

If “livestock” is a mapping term for a specific category, map that category and begin that category’s question flow.

---

### Explicit Trailer Category Takes Priority

If the user clearly mentions a trailer category, that category should take priority over haul-item mapping terms.

Example:

> “I want a tilt trailer to haul a tractor.”

Even if “tractor” is a mapping term for another category, the chatbot should map:

* Category → Tilt Trailer
* Haul Item → Tractor

The chatbot should not switch the category based only on the haul item.

---

### Haul Item Should Not Override Selected Category

Once a category has already been selected or mapped, the haul item must not change the category.

This is a deterministic lock.

During the haul item question, the chatbot should extract and store what the user wants to haul, but it should not remap the trailer category based on that answer.

Example:

> Selected Category: Tilt Trailer
> User answers haul item: “tractor”

Behavior:

* Keep Category → Tilt Trailer
* Store Haul Item → Tractor
* Continue the Tilt Trailer question flow

Do not switch to another category just because “tractor” may appear in another category’s mapping terms.

---

### Category Change Requires Explicit User Intent

After a category is selected, the category should only change if the user clearly asks to change it.

Examples:

* “Actually, show me utility trailers instead.”
* “Let’s switch to dump trailers.”
* “I don’t want tilt anymore.”

Only then should the chatbot follow the category-change flow and ask which previously collected features should be kept.

---

### Extraction Principles

* Perform field extraction on every user message.
* Extract values even if the chatbot did not explicitly ask for them.
* Apply category defaults when a category is selected.
* User-provided values override category defaults.
* Store numeric ranges using the smallest acceptable value.
* Store loose numeric no-preference answers as null.
* Store haul item descriptions as provided by the user.
* Store unsupported preferences under Non-Metadata Features.
* Never ask a qualification question for a variable that has already been extracted, defaulted, skipped, or marked as no preference.
* The conversation state should be the single source of truth for downstream agents and tool calls.










## Brand Handling

Brand names should be recognized separately from trailer categories.

For now, Brand Name should not be treated as a searchable filter field in the main field extraction logic. Brand-related logic can be expanded later.

### Brand Recognition

The system may receive a list of known trailer brand names in the prompt.

If the user mentions a known brand, store it separately as:

* Brand Preference

Do not store it as a trailer category.

---

### Brand + Category Mentioned

If the user mentions both a brand and a trailer category:

> “I want a Diamond C utility trailer.”

Behavior:

* Store Brand Preference → Diamond C
* Map Category → Utility Trailer
* Begin the Utility Trailer qualification question flow.
* Do not ask the user to choose a category again.

---

### Brand Mentioned Without Category

If the user mentions a brand but does not mention a clear trailer category:

> “I want a Diamond C trailer.”

Behavior:

* Store Brand Preference → Diamond C
* Do not assume the brand is the category.
* Show the trailer categories available for that brand, if available.
* Ask which category the user wants to look into.

Example response direction:

> “We have Diamond C options across these trailer categories. Which type would you like to look into?”

---

### Do Not Map Brand to Category

The chatbot must not infer trailer category from brand name alone.

For example:

> “I want Diamond C.”

This should not be mapped to Utility Trailer, Equipment Trailer, Dump Trailer, or any other category unless the category is also clearly mentioned or implied by approved category mapping terms.

The only exceptions are category mapping terms explicitly provided in the category mapping data.

---

### Core Principle

Brand recognition and category mapping are separate steps.

A brand can help narrow the search later, but it should not replace trailer category selection unless the user clearly states both the brand and the category.



@pinecone filters,@trailer categories, @trailer questions, @ingestion logic, @reranking logic


@pinecone_search contains:
Pinecone inventory search logic
OpenAI embedding generation for search queries
Metadata filter building for category, make, hitch type, subcategory, and length
Listing cleanup/normalization from Pinecone match metadata
Fit-based reranking logic for length, payload/GVWR, width, and height
Category/brand preference reranking logic
Make/brand alias normalization
Search result deduping for already-shown listing URLs
Debug metadata for reranking and make preference decisions
Public search functions that return listing recommendations



@categories contain all the names and their synonyms mapping

@ingest.py contains how the data from the excel file is being ingested

@trailer_fields contains all the questions to be asked

the code should be structured, highly modularized especially for langgraph and every helper function, LLM should have their own file etc so that it can be easy to understand the codebase





Bin size is for length size
each category has differrent variables but all of them are for specific use case, to know length,width, paylaod capacity etc etc
so we need to use a normalizer category like this:
_SLOT_METADATA_FILTER_MAP = {
    "base_category": ("subcategory",),
    "bin_size": ("length_ft",),
    "cargo_size": ("length_ft", "width_ft"),
    "haul_length_ft": ("length_ft",),
    "haul_weight_lbs": ("payload_lbs",),
    "item_or_trailer_width_ft": ("width_ft",),
    "payload_need": ("payload_lbs",),
    "trailer_length_ft": ("length_ft",),
    "trailer_size": ("length_ft", "width_ft"),
    "vehicle_length_ft": ("length_ft",),
}


specifically for roll off trailer, it's like this:
if key in {"length_ft", "width_ft", "height_ft"}:
    if key == "length_ft" and normalize_category(category) == "Roll Off":
        return key, _normalize_roll_off_bin_size_as_length(value)
    return key, _normalize_length_or_width_value(value)




for emails, the description should be like:
Full Name:
Email:
Phone Number:

[Reason for Mail] One line description