Classify complexity. ONE word: super_easy, easy, medium, hard, super_hard

super_easy: hi, hey, thanks, ok, yes, no, bye
easy: remind me, what is, how much, quick question
medium: write code, function, send email, draft, research, fix bug
hard: refactor, debug crash, multi-file change
super_hard: design, distributed, system architecture, prove theorem, autonomous

RULE: "design" in message = super_hard (not hard)

Examples:
"Hey" -> super_easy
"What is 2+2?" -> easy
"Write a sort function" -> medium
"Send email to Bob" -> medium
"Refactor the auth module" -> hard
"Refactor to microservices" -> hard
"Design a system" -> super_hard
"Design a distributed system architecture" -> super_hard

Message: {MESSAGE}

Complexity:
