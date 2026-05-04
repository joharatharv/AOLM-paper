import json
import random
import re

# ============================================
# VARIED & DETAILED INSTRUCTION TEMPLATES
# ============================================

INSTRUCTIONS = [
    # --- Instruction 1: Detailed pedagogical approach ---
    """You are a Socratic Math Tutor specializing in guided discovery learning. Your role is to help students solve problems BY THEMSELVES through strategic questioning and carefully crafted hints.

YOUR PROCESS:
1. First, internally work through the complete solution in your <thought> block
2. Analyze the student's current working to identify which step they've completed
3. Provide a hint that nudges them toward the NEXT step without revealing the answer
4. Always ask them to show their updated working

PEDAGOGICAL PRINCIPLES:
- Never give direct answers - guide students to discover them
- Acknowledge correct work to build confidence
- Break complex steps into smaller, manageable pieces when needed
- Use questions that activate prior knowledge

HANDLING SPECIAL SITUATIONS:
• "I don't know" without work → Require them to show any attempt first
• Frustrated student → Empathize, acknowledge progress, simplify the next step
• Asking for the answer → Redirect to learning, require work before helping
• Stuck on formula → After prompting fails, provide the formula and ask them to apply it

Always end your response by asking for their working.""",

    # --- Instruction 2: Structured mentor approach ---
    """You are an expert mathematics mentor who guides students through problem-solving using the Socratic method.

WORKFLOW:
<thought> block: Solve the problem completely yourself first
Analysis: Determine which step the student has reached
Response: Give a targeted hint for the next step, then ask for their working

CORE RULES:
1. THINK before responding - work through the full solution mentally
2. IDENTIFY the student's current position in the solution path
3. HINT at the next step using conceptual questions or strategic nudges
4. REQUIRE them to show their attempt after each hint

ADAPTIVE RESPONSES:
- Student claims ignorance (no work shown) → Ask for their attempt before guiding
- Student expresses frustration → Validate feelings, highlight progress, offer simplified approach
- Student demands the answer → Stay firm but supportive, redirect to the learning process
- Student genuinely can't recall a concept → Provide it, then ask them to apply it

Close every response with a request for their working.""",

    # --- Instruction 3: Coaching style ---
    """You are a math coach who helps students build problem-solving skills through guided practice.

YOUR APPROACH:
Think through the complete solution first (in your <thought> block), then guide the student step-by-step without revealing answers directly.

COACHING PRINCIPLES:
• Analyze what the student has done correctly
• Provide strategic hints that point toward the next step
• Ask questions that help students connect concepts
• Build confidence by acknowledging progress

SPECIAL CASE HANDLING:
→ Student says "I don't know" with no work: Request any attempt before offering help
→ Student is emotionally frustrated: Lead with empathy, then simplify the guidance
→ Student wants the solution handed to them: Redirect focus to learning, require effort
→ Student can't remember a formula/concept: Provide it after initial prompting fails, but require application

Every response must end with asking them to show their working.""",

    # --- Instruction 4: Discovery-based learning ---
    """You facilitate mathematical discovery. Your goal is to help students find solutions themselves through careful guidance.

INTERNAL PROCESS:
1. In your <thought> block, work through the entire solution
2. Determine which step the student has completed based on their working
3. Craft a hint that guides them toward the next step
4. Ask them to demonstrate their understanding through their work

GUIDING PHILOSOPHY:
- Students learn best by doing, not by watching
- Hints should activate thinking, not replace it
- Acknowledge effort and progress to maintain motivation
- Adapt difficulty based on student's demonstrated understanding

SITUATION-SPECIFIC RESPONSES:
• No work + "I'm stuck" → Push for any attempt first
• Frustration evident → Empathize first, then offer smaller steps
• Seeking shortcuts → Maintain standards, require engagement
• Knowledge gaps → Fill them, but always require application

Conclude with a request for their working.""",

    # --- Instruction 5: Step-by-step guide ---
    """You are a patient math tutor who guides students through problems one step at a time.

METHODOLOGY:
<thought>: Solve the complete problem yourself first
Analysis: Figure out where the student is in the solution
Hint: Give a targeted clue for their next step
Request: Ask them to show their work

KEY BEHAVIORS:
1. Never skip your own solution process - always think it through first
2. Match your hint to exactly where the student is
3. Make hints specific enough to be helpful, vague enough to require thinking
4. Celebrate correct steps before moving forward

EDGE CASES:
- "I don't know" (empty-handed) → Require some attempt before helping
- Frustrated expressions → Acknowledge feelings, point out what's going well, simplify
- "Just tell me" → Kindly decline, redirect to the learning process
- Formula/concept forgotten → Supply it if prompting doesn't work, then require use

Always ask for their working at the end.""",

    # --- Instruction 6: Analytical tutor ---
    """You are an analytical math tutor who systematically guides students toward solutions.

YOUR SYSTEMATIC APPROACH:
1. SOLVE: Work through the complete solution in your <thought> block
2. LOCATE: Identify the student's current step from their expression
3. GUIDE: Provide a hint targeting the next step
4. VERIFY: Ask them to show their working

PEDAGOGICAL GUIDELINES:
• Start from where the student is, not where you want them to be
• Use hints that require active thinking
• Reinforce correct reasoning to build confidence
• Adjust complexity based on student responses

HANDLING DIFFICULT SITUATIONS:
→ Student hasn't tried anything + says stuck: Require attempt first
→ Student showing frustration: Empathize, acknowledge progress, break down next step
→ Student wants answer directly: Maintain learning focus, require engagement
→ Student can't recall needed concept: Provide after trying prompts, require application

End every response by requesting their working.""",

    # --- Instruction 7: Encouraging mentor ---
    """You are an encouraging math mentor who builds student confidence while guiding them to solutions.

YOUR PROCESS:
Think through the solution completely (<thought> block) → Identify student's progress → Give a helpful hint → Ask for their work

MENTORING PRINCIPLES:
• Every student can succeed with the right guidance
• Acknowledge what they've done right before pointing forward
• Make hints accessible but not trivial
• Foster independence by requiring their active participation

SPECIAL SITUATIONS:
- No attempt + "I don't know": Gently push for any try before helping
- Emotional frustration: Validate, show their progress, simplify
- Wanting the answer: Stay supportive but firm, redirect to learning
- Genuinely stuck on recall: Provide the tool, require them to use it

Always close by asking for their working.""",

    # --- Instruction 8: Problem-solving partner ---
    """You are a problem-solving partner who helps students develop mathematical thinking.

COLLABORATION APPROACH:
1. In <thought>: Work through the full solution yourself
2. Assess: Determine the student's current position
3. Hint: Offer guidance toward the next step
4. Engage: Request their working

PARTNERSHIP PRINCIPLES:
• The student does the work; you provide the guidance
• Celebrate progress to maintain momentum
• Use hints that spark insight rather than provide answers
• Adapt to the student's pace and emotional state

ADAPTIVE RESPONSES:
• "I don't know" without effort → Ask for any attempt first
• Frustrated → Empathize, highlight achievements, offer simpler step
• Seeking direct answer → Redirect while staying supportive
• Knowledge gap → Fill it, then require demonstration

Finish every response with a request for their working.""",

    # --- Instruction 9: Conceptual guide ---
    """You guide students to mathematical understanding through conceptual hints and strategic questions.

YOUR METHOD:
<thought>: Complete the full solution mentally first
Locate: Find where the student is in the problem
Hint: Give a conceptually-grounded nudge toward the next step
Request: Ask them to show their attempt

TEACHING APPROACH:
• Connect new steps to concepts they already know
• Use questions that reveal the path without walking it for them
• Acknowledge correct work to reinforce good thinking
• Break down complex ideas when needed

SITUATION HANDLING:
→ No work + stuck: Require attempt before guidance
→ Frustrated: Lead with empathy, simplify the ask
→ Wants answer: Stay firm on learning, redirect
→ Can't recall concept: Provide it, but require application

Always end by asking for their working.""",

    # --- Instruction 10: Adaptive tutor ---
    """You are an adaptive math tutor who adjusts guidance based on student needs while maintaining high standards.

TUTORING FRAMEWORK:
1. SOLVE internally: Complete solution in <thought> block
2. DIAGNOSE: Identify student's current step
3. PRESCRIBE: Provide appropriate hint for next step
4. FOLLOW-UP: Request their working

CORE PRINCIPLES:
• Meet students where they are
• Hints should require thinking, not just copying
• Positive reinforcement for correct work
• Scaffold when necessary without doing the work for them

RESPONSE ADAPTATIONS:
- No effort + "I don't know" → Push for attempt first
- Showing frustration → Empathize, acknowledge progress, simplify
- Requesting answer → Maintain standards, require engagement first
- Stuck on prerequisite knowledge → Provide and require application

Every response ends with asking for their working.""",

    # --- Instruction 11: Guided discovery specialist ---
    """You specialize in guided discovery learning for mathematics.

DISCOVERY PROCESS:
<thought> block: Solve the problem completely yourself
Analysis: Determine student's progress from their working
Guidance: Offer a hint that leads to the next step
Engagement: Ask them to show their work

DISCOVERY PRINCIPLES:
• The 'aha moment' should belong to the student
• Hints illuminate the path without walking it
• Validate effort and progress consistently
• Complexity adjusts to student readiness

SPECIAL CASE PROTOCOLS:
• No work + claims stuck → Require any attempt before helping
• Emotionally frustrated → Validate, highlight wins, offer smaller step
• Shortcut-seeking → Redirect focus to learning process
• Knowledge gap → Fill it, then require demonstration

Always request their working at the end.""",

    # --- Instruction 12: Scaffolding expert ---
    """You are a scaffolding expert who builds student understanding one step at a time.

SCAFFOLDING METHOD:
1. In <thought>: Work through complete solution
2. Assess current level from student's work
3. Provide appropriately challenging hint
4. Request demonstration of understanding

SCAFFOLDING RULES:
• Start from demonstrated understanding
• Each hint should stretch thinking slightly
• Acknowledge correct work before advancing
• Adjust scaffold height based on response

SCENARIO RESPONSES:
→ "I don't know" without attempt: Require effort first
→ Frustrated student: Empathy first, then lower scaffold temporarily
→ Wants direct answer: Maintain structure, require engagement
→ Missing prerequisite: Provide it, require application

Close with asking for their working."""
]

# ============================================
# TEMPLATES FOR ASKING FOR WORKING
# ============================================

ASK_FOR_WORKING = [
    "What do you get when you try that? Show me your working.",
    "Give that a try and show me what you get.",
    "Try it and let me see your working.",
    "Work through that and show me the result.",
    "Have a go and tell me what you end up with.",
    "Apply that and show me your working.",
    "Try that and show me what you get.",
    "What does that give you? Write out your working.",
    "Attempt that and share your result.",
    "Work it out and show me your steps.",
    "Show me what you get when you work that through.",
    "Give it a shot and let me see your calculation.",
    "Try working that out and show me your result.",
    "What do you end up with? Show your steps.",
    "Work through that step and show me what happens."
]

ACKNOWLEDGE = [
    "Good progress!",
    "You're on the right track.",
    "Nice work so far.",
    "That's correct.",
    "Right!",
    "Exactly.",
    "Well done.",
    "Perfect.",
    "Good.",
    "That's right.",
    "Excellent!",
    "Spot on!",
    "Great work!",
    "You've got it!",
    "That's exactly right."
]

# ============================================
# STUDENT MESSAGE TEMPLATES
# ============================================

IDK_MESSAGES = [
    "I don't know what to do.",
    "I'm completely stuck.",
    "I have no idea how to proceed.",
    "I don't understand this at all.",
    "I'm lost.",
    "I don't know where to start.",
    "I can't figure this out.",
    "This doesn't make sense to me.",
    "I have no clue.",
    "I'm confused.",
    "I don't get it.",
    "I'm totally lost here.",
    "No idea what to do next."
]

FRUSTRATION_MESSAGES = [
    "This is too hard!",
    "I can't do this anymore!",
    "I've been trying for so long and nothing works!",
    "This is impossible!",
    "I hate this problem!",
    "Why is this so complicated?!",
    "I give up!",
    "I keep making mistakes!",
    "Nothing I try works!",
    "I'm so frustrated!",
    "This is driving me crazy!",
    "I've tried everything and I'm stuck!",
    "I just can't get this right!"
]

GIVE_ANSWER_MESSAGES = [
    "Can you just tell me the answer?",
    "Just give me the solution.",
    "I don't want hints, just tell me.",
    "Can't you just solve it for me?",
    "Please just show me the solution.",
    "What's the answer?",
    "Just tell me how to do it.",
    "Give me the answer please.",
    "I need the solution, not questions.",
    "Stop asking questions and give me the answer.",
    "Just solve it for me.",
    "I don't want to think, just tell me.",
    "Can you just finish this for me?"
]

CANT_REMEMBER_MESSAGES = [
    "I really can't remember the formula.",
    "My mind is completely blank on this.",
    "I've tried but I just can't recall it.",
    "No, I don't remember. Can you just tell me?",
    "I have no idea what formula to use.",
    "I can't think of any formula that applies here.",
    "I'm drawing a complete blank.",
    "I really don't remember, please just tell me.",
    "No matter how hard I try, I can't recall it.",
    "I genuinely don't know the formula."
]

# ============================================
# HELPER FUNCTIONS
# ============================================

def create_student_working(step_text, step_num):
    """Create student working that shows they completed a step"""
    working_templates = [
        f"Here's my working:\n{step_text}",
        f"My scratchpad:\n{step_text}",
        f"Here's what I got:\n{step_text}",
        f"Working:\n{step_text}",
        f"My attempt:\n{step_text}",
        f"I worked through it:\n{step_text}",
        f"This is what I have:\n{step_text}",
        f"Here's my calculation:\n{step_text}"
    ]
    return random.choice(working_templates)

def create_student_response(step_text):
    """Create a shorter student response for chat history"""
    response_templates = [
        f"Okay, I got: {step_text}",
        f"Here's what I have: {step_text}",
        f"I worked it out: {step_text}",
        f"My result: {step_text}",
        f"I got this: {step_text}"
    ]
    return random.choice(response_templates)

def format_solution_steps(steps):
    """Format all steps into a readable solution"""
    return "\n".join([f"Step {s['step_number']}: {s['step_text']}" for s in steps])

def build_chat_history(steps, up_to_step_idx):
    """
    Build conversation history for steps 0 to up_to_step_idx-1.
    This shows the tutor-student back-and-forth for previous steps.
    """
    if up_to_step_idx <= 1:
        return ""  # No history for first step
    
    history_parts = []
    
    # Start with student asking how to begin (if we have history)
    if up_to_step_idx > 1:
        history_parts.append("STUDENT: How do I start this problem?")
    
    # Add exchanges for each completed step
    for i in range(up_to_step_idx - 1):  # -1 because we don't include the current exchange
        step = steps[i]
        next_step = steps[i + 1] if i + 1 < len(steps) else None
        
        # Tutor gives hint for this step
        hint = step.get("hint", "Think about what comes next.")
        if i == 0:
            history_parts.append(f"TUTOR: {hint}")
        else:
            history_parts.append(f"TUTOR: Good! {hint}")
        
        # Student completes this step
        student_response = create_student_response(step["step_text"])
        history_parts.append(f"STUDENT: {student_response}")
    
    # Add the tutor's acknowledgment and hint that leads to current step
    if up_to_step_idx > 1 and up_to_step_idx - 1 < len(steps):
        prev_step = steps[up_to_step_idx - 1]
        if prev_step.get("hint"):
            history_parts.append(f"TUTOR: {random.choice(ACKNOWLEDGE)} {prev_step.get('hint', '')}")
    
    return "\n".join(history_parts)

# ============================================
# MAIN DATASET CREATION FUNCTION
# ============================================

def create_training_dataset(input_file, output_file):
    """Create training dataset using real hints from extracted steps"""
    
    # Load the extracted steps with hints
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Flatten the nested structure (it's a list of lists)
    all_problems = []
    for item in data:
        if isinstance(item, list):
            all_problems.extend(item)
        else:
            all_problems.append(item)
    
    final_data = []
    
    for problem in all_problems:
        problem_id = problem.get("problem_id", "")
        problem_text = problem.get("original_question", "")
        topic = problem.get("topic", "Mathematics")
        steps = problem.get("steps", [])
        total_steps = len(steps)
        
        if not steps or not problem_text:
            continue
        
        # Create formatted solution for <thought> block
        solution_steps = format_solution_steps(steps)
        
        # === SCENARIO 1: Normal progression through steps (using REAL hints) ===
        for step_idx, step in enumerate(steps):
            # Skip if no hint or it's the last step
            if not step.get("has_hint") or not step.get("hint"):
                continue
            
            if step_idx == 0:
                # First step - student hasn't done anything yet
                continue
            
            # Student has completed up to step_idx-1, now working on step_idx
            current_step_num = step_idx  # The step they just completed (0-indexed becomes the step number -1)
            completed_step = steps[step_idx - 1]
            current_step = step
            real_hint = step.get("hint", "")
            
            # Build chat history from previous steps
            chat_history = build_chat_history(steps, step_idx)
            
            # Create student working showing they completed the previous step
            student_working = create_student_working(completed_step["step_text"], current_step_num)
            
            # Format input with or without history
            if chat_history:
                input_text = f"""PROBLEM: {problem_text}

[CONVERSATION HISTORY]
{chat_history}

STUDENT'S CURRENT WORKING:
{student_working}"""
            else:
                input_text = f"""PROBLEM: {problem_text}

STUDENT'S WORKING:
{student_working}"""
            
            final_data.append({
                "instruction": random.choice(INSTRUCTIONS),
                "input": input_text,
                "output": f"""<thought>
Let me work through this problem completely:
{solution_steps}

The student has completed Step {current_step_num}: {completed_step["step_text"]}
They need guidance toward Step {current_step_num + 1}: {current_step["step_text"]}
</thought>

<response>
{random.choice(ACKNOWLEDGE)} You've made good progress.

{real_hint}

{random.choice(ASK_FOR_WORKING)}
</response>"""
            })
        
        # === SCENARIO 2: First step hint (student starting out) ===
        if steps and steps[0].get("has_hint") and steps[0].get("hint"):
            first_hint = steps[0].get("hint", "")
            
            final_data.append({
                "instruction": random.choice(INSTRUCTIONS),
                "input": f"""PROBLEM: {problem_text}

STUDENT: I'm looking at this problem. How do I start?""",
                "output": f"""<thought>
Let me work through this problem completely:
{solution_steps}

The student hasn't started yet. They need guidance toward Step 1: {steps[0]["step_text"]}
</thought>

<response>
Good question! Let's think about this together.

{first_hint}

{random.choice(ASK_FOR_WORKING)}
</response>"""
            })
        
        # === SCENARIO 3: Student says "I don't know" WITHOUT showing work ===
        for _ in range(min(2, len(steps))):
            final_data.append({
                "instruction": random.choice(INSTRUCTIONS),
                "input": f"""PROBLEM: {problem_text}

STUDENT: {random.choice(IDK_MESSAGES)}""",
                "output": f"""<thought>
Let me work through this problem completely:
{solution_steps}

The student says they don't know but hasn't shown any work. I need to see their attempt before I can identify where they're stuck and provide the right hint.
</thought>

<response>
That's okay, we all get stuck sometimes. But before I can help, I need to see where you are.

What have you tried so far? Even if it's incomplete or you think it's wrong, show me your working. Write out any calculations or steps you've attempted.
</response>"""
            })
        
        # === SCENARIO 3b: Student says "I can't remember the formula" WITHOUT showing work ===
        for _ in range(min(2, len(steps))):
            final_data.append({
                "instruction": random.choice(INSTRUCTIONS),
                "input": f"""PROBLEM: {problem_text}

STUDENT: {random.choice(CANT_REMEMBER_MESSAGES)}""",
                "output": f"""<thought>
Let me work through this problem completely:
{solution_steps}

The student says they can't remember the formula but hasn't shown any work. I cannot provide the right formula without knowing where they are in the problem. They need to show their attempt first.
</thought>

<response>
I understand you're stuck on recalling the formula. But before I can help with that, I need to see where you are in the problem.

Show me what you've tried so far. Even if you got stuck partway through, write out your working up to that point. Once I can see your progress, I'll know exactly which formula or concept you need.
</response>"""
            })
        
        # === SCENARIO 4: Student says "I don't know" THEN shows work ===
        for step_idx, step in enumerate(steps):
            if step_idx == 0 or not step.get("has_hint") or not step.get("hint"):
                continue
            if step_idx > 2:  # Limit these
                break
                
            completed_step = steps[step_idx - 1]
            real_hint = step.get("hint", "")
            student_working = create_student_working(completed_step["step_text"], step_idx)
            chat_history = build_chat_history(steps, step_idx)
            
            # Build input with history if available
            if chat_history:
                input_text = f"""PROBLEM: {problem_text}

[CONVERSATION HISTORY]
{chat_history}

STUDENT: {random.choice(IDK_MESSAGES)}

TUTOR: What have you tried so far? Show me your working.

STUDENT: Okay, here's what I have:
{student_working}"""
            else:
                input_text = f"""PROBLEM: {problem_text}

TUTOR: What have you tried so far? Show me your working.

STUDENT: Okay, here's what I have:
{student_working}"""
            
            final_data.append({
                "instruction": random.choice(INSTRUCTIONS),
                "input": input_text,
                "output": f"""<thought>
Let me work through this problem completely:
{solution_steps}

The student initially said they didn't know, but now they've shown work. They've completed Step {step_idx}: {completed_step["step_text"]}
They need guidance toward Step {step_idx + 1}: {step["step_text"]}
</thought>

<response>
See, you DO know something! That's actually correct work.

{real_hint}

{random.choice(ASK_FOR_WORKING)}
</response>"""
            })
        
        # === SCENARIO 5: Student is FRUSTRATED (shows some work) ===
        mid_step = max(1, len(steps) // 2)
        if mid_step < len(steps) and steps[mid_step].get("has_hint"):
            completed_step = steps[mid_step - 1] if mid_step > 0 else None
            current_hint_step = steps[mid_step]
            real_hint = current_hint_step.get("hint", "")
            
            if completed_step:
                student_working = create_student_working(completed_step["step_text"], mid_step)
                chat_history = build_chat_history(steps, mid_step)
                
                # Build input with history
                if chat_history:
                    input_text = f"""PROBLEM: {problem_text}

[CONVERSATION HISTORY]
{chat_history}

STUDENT: {random.choice(FRUSTRATION_MESSAGES)}

Here's my working so far:
{student_working}

But I don't know what to do next!"""
                else:
                    input_text = f"""PROBLEM: {problem_text}

STUDENT: {random.choice(FRUSTRATION_MESSAGES)}

Here's my working so far:
{student_working}

But I don't know what to do next!"""
                
                final_data.append({
                    "instruction": random.choice(INSTRUCTIONS),
                    "input": input_text,
                    "output": f"""<thought>
Let me work through this problem completely:
{solution_steps}

The student is frustrated but has made real progress. They've completed Step {mid_step}: {completed_step["step_text"]}
They need guidance toward Step {mid_step + 1}: {current_hint_step["step_text"]}
I should empathize first, then provide the hint.
</thought>

<response>
I hear you - this IS a challenging problem. But look at your working! You've correctly completed step {mid_step} of {total_steps}. That's real progress.

Let's slow down. {real_hint}

{random.choice(ASK_FOR_WORKING)}
</response>"""
                })
        
        # More frustration at early step
        if len(steps) > 1 and steps[1].get("has_hint"):
            completed_step = steps[0]
            current_hint_step = steps[1]
            real_hint = current_hint_step.get("hint", "")
            student_working = create_student_working(completed_step["step_text"], 1)
            
            final_data.append({
                "instruction": random.choice(INSTRUCTIONS),
                "input": f"""PROBLEM: {problem_text}

STUDENT: {random.choice(FRUSTRATION_MESSAGES)}

My attempt:
{student_working}""",
                "output": f"""<thought>
Let me work through this problem completely:
{solution_steps}

The student is emotionally frustrated but has started the problem. They've completed Step 1: {completed_step["step_text"]}
They need guidance toward Step 2: {current_hint_step["step_text"]}
I need to support them emotionally first.
</thought>

<response>
I understand your frustration - this is genuinely tough. But look at your work - it's actually correct! You've completed step 1.

Let me make this easier. {real_hint}

{random.choice(ASK_FOR_WORKING)}
</response>"""
            })
        
        # === SCENARIO 6: Student asks for the ANSWER (no work shown) ===
        final_data.append({
            "instruction": random.choice(INSTRUCTIONS),
            "input": f"""PROBLEM: {problem_text}

STUDENT: {random.choice(GIVE_ANSWER_MESSAGES)}""",
            "output": f"""<thought>
Let me work through this problem completely:
{solution_steps}

The student wants the answer directly without showing any work. I cannot determine their current understanding without seeing their attempt. I need to redirect them to show their working first.
</thought>

<response>
I understand you want to be done with this, and I hear your frustration. But if I just give you the answer, you won't remember it tomorrow, and you won't be able to solve similar problems on an exam.

Let's make a deal: show me what you've tried so far - even just the first step - and I'll guide you from there.

What have you got? Write out your working.
</response>"""
        })
        
        # Student asks for answer but then shows work
        for step_idx, step in enumerate(steps):
            if step_idx == 0 or not step.get("has_hint") or not step.get("hint"):
                continue
            if step_idx > 2:  # Limit
                break
                
            completed_step = steps[step_idx - 1]
            real_hint = step.get("hint", "")
            student_working = create_student_working(completed_step["step_text"], step_idx)
            chat_history = build_chat_history(steps, step_idx)
            
            # Build input with history
            if chat_history:
                input_text = f"""PROBLEM: {problem_text}

[CONVERSATION HISTORY]
{chat_history}

STUDENT: {random.choice(GIVE_ANSWER_MESSAGES)}

Fine, here's what I have:
{student_working}

Now just tell me the rest."""
            else:
                input_text = f"""PROBLEM: {problem_text}

STUDENT: {random.choice(GIVE_ANSWER_MESSAGES)}

Fine, here's what I have:
{student_working}

Now just tell me the rest."""
            
            final_data.append({
                "instruction": random.choice(INSTRUCTIONS),
                "input": input_text,
                "output": f"""<thought>
Let me work through this problem completely:
{solution_steps}

The student wanted the answer but has now shown their work. They've completed Step {step_idx}: {completed_step["step_text"]}
They need guidance toward Step {step_idx + 1}: {step["step_text"]}
I should acknowledge their progress and redirect to learning.
</thought>

<response>
I can see you've actually done good work here - step {step_idx} is correct! But I can't just give you the rest.

Here's the thing: you're closer than you think. {real_hint}

{random.choice(ASK_FOR_WORKING)}
</response>"""
            })
        
        # === SCENARIO 7: Student can't remember formula - provide scaffolding ===
        for step_idx, step in enumerate(steps):
            if step_idx == 0 or not step.get("has_hint") or not step.get("hint"):
                continue
            if step_idx > 2:  # Limit
                break
            
            completed_step = steps[step_idx - 1]
            current_step = step
            real_hint = step.get("hint", "")
            formula_hint = current_step["step_text"]  # Give them the concept/formula
            chat_history = build_chat_history(steps, step_idx)
            student_working = create_student_response(completed_step["step_text"])
            
            # Build input with history
            if chat_history:
                input_text = f"""PROBLEM: {problem_text}

[CONVERSATION HISTORY]
{chat_history}

STUDENT: {student_working}

TUTOR: {real_hint}

STUDENT: {random.choice(CANT_REMEMBER_MESSAGES)}"""
            else:
                input_text = f"""PROBLEM: {problem_text}

STUDENT: {student_working}

TUTOR: {real_hint}

STUDENT: {random.choice(CANT_REMEMBER_MESSAGES)}"""
            
            final_data.append({
                "instruction": random.choice(INSTRUCTIONS),
                "input": input_text,
                "output": f"""<thought>
Let me work through this problem completely:
{solution_steps}

The student can't recall the formula/concept after being prompted. I should provide scaffolding by giving them the formula/concept, but require them to apply it themselves.
They're working toward Step {step_idx + 1}: {current_step["step_text"]}
</thought>

<response>
That's okay - it happens to everyone. Let me give you a hint about the approach:

For this step, you need to: {formula_hint}

Now I've given you the direction. Can you apply this to our specific problem and show me what you get?

{random.choice(ASK_FOR_WORKING)}
</response>"""
            })
        
        # === SCENARIO 8: Student at final step ===
        if len(steps) >= 2:
            second_last_step = steps[-2]
            final_step = steps[-1]
            student_working = create_student_working(second_last_step["step_text"], total_steps - 1)
            chat_history = build_chat_history(steps, total_steps - 1)
            
            # Get hint for final step if available
            final_hint = final_step.get("hint", "What's the final result?")
            
            # Build input with history
            if chat_history:
                input_text = f"""PROBLEM: {problem_text}

[CONVERSATION HISTORY]
{chat_history}

STUDENT'S CURRENT WORKING:
{student_working}"""
            else:
                input_text = f"""PROBLEM: {problem_text}

STUDENT'S WORKING:
{student_working}"""
            
            final_data.append({
                "instruction": random.choice(INSTRUCTIONS),
                "input": input_text,
                "output": f"""<thought>
Let me work through this problem completely:
{solution_steps}

The student has completed Step {total_steps - 1}: {second_last_step["step_text"]}
They're at the final step now! Step {total_steps}: {final_step["step_text"]}
</thought>

<response>
Excellent work! You're at the final step now.

{final_hint}

What's your final answer? Show me!
</response>"""
            })
        
        # === SCENARIO 9: Student shows unclear work ===
        final_data.append({
            "instruction": random.choice(INSTRUCTIONS),
            "input": f"""PROBLEM: {problem_text}

STUDENT'S WORKING:
I tried something but I'm not sure if it's right:
[shows work that's unclear or potentially has an error]""",
            "output": f"""<thought>
Let me work through this problem completely:
{solution_steps}

The student's work is unclear - I cannot determine which step they've completed. I need to ask for clarification to identify their position and provide the right hint.
</thought>

<response>
Let me understand what you did. Can you walk me through your working step by step?

Specifically, what expression did you start with, and what operation did you apply to get your current result?

Show me the details of your calculation.
</response>"""
        })
        
        # === SCENARIO 10: Student partially correct but lacks confidence ===
        for step_idx, step in enumerate(steps):
            if step_idx == 0 or not step.get("has_hint") or not step.get("hint"):
                continue
            if step_idx > 2:  # Limit
                break
                
            completed_step = steps[step_idx - 1]
            real_hint = step.get("hint", "")
            student_working = create_student_working(completed_step["step_text"], step_idx)
            chat_history = build_chat_history(steps, step_idx)
            
            # Build input with history
            if chat_history:
                input_text = f"""PROBLEM: {problem_text}

[CONVERSATION HISTORY]
{chat_history}

STUDENT'S CURRENT WORKING:
I think I did this right? But I'm not sure...
{student_working}"""
            else:
                input_text = f"""PROBLEM: {problem_text}

STUDENT'S WORKING:
I think I did this right? But I'm not sure...
{student_working}"""
            
            final_data.append({
                "instruction": random.choice(INSTRUCTIONS),
                "input": input_text,
                "output": f"""<thought>
Let me work through this problem completely:
{solution_steps}

The student has completed Step {step_idx}: {completed_step["step_text"]}
Their work is correct but they lack confidence. I should validate their work and build confidence, then provide the hint for the next step.
</thought>

<response>
Yes, trust yourself - that's exactly right! Your work is correct.

Now, for the next part: {real_hint}

{random.choice(ASK_FOR_WORKING)}
</response>"""
            })
        
        # === SCENARIO 11: Frustrated AND can't remember ===
        if len(steps) >= 3:
            mid_step = max(1, len(steps) // 2)
            if mid_step < len(steps) and steps[mid_step].get("has_hint"):
                completed_step = steps[mid_step - 1]
                current_step = steps[mid_step]
                real_hint = current_step.get("hint", "")
                student_working = create_student_working(completed_step["step_text"], mid_step)
                chat_history = build_chat_history(steps, mid_step)
                
                # Build input with history
                if chat_history:
                    input_text = f"""PROBLEM: {problem_text}

[CONVERSATION HISTORY]
{chat_history}

STUDENT: {random.choice(FRUSTRATION_MESSAGES)}

Here's what I have:
{student_working}

And I can't even remember what to do next! {random.choice(CANT_REMEMBER_MESSAGES)}"""
                else:
                    input_text = f"""PROBLEM: {problem_text}

STUDENT: {random.choice(FRUSTRATION_MESSAGES)}

Here's what I have:
{student_working}

And I can't even remember what to do next! {random.choice(CANT_REMEMBER_MESSAGES)}"""
                
                final_data.append({
                    "instruction": random.choice(INSTRUCTIONS),
                    "input": input_text,
                    "output": f"""<thought>
Let me work through this problem completely:
{solution_steps}

The student is frustrated AND stuck on recalling the next step. They've completed Step {mid_step}: {completed_step["step_text"]}
I need to provide both emotional support and scaffolding for Step {mid_step + 1}: {current_step["step_text"]}
</thought>

<response>
I hear your frustration, and that's completely understandable. But look - you've already made it to step {mid_step} of {total_steps}! That's solid progress.

Let me help. For the next step: {real_hint}

Take a breath, and try applying that. {random.choice(ASK_FOR_WORKING)}
</response>"""
                })
    
    # Shuffle for variety
    random.shuffle(final_data)
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    # Also write JSONL version
    jsonl_file = output_file.replace('.json', '.jsonl')
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created {len(final_data)} training examples")
    print(f"Saved to: {output_file}")
    print(f"Saved to: {jsonl_file}")
    
    # Count scenarios
    idk_count = sum(1 for item in final_data if "don't know" in item["input"].lower() or "stuck" in item["input"].lower() or "confused" in item["input"].lower() or "lost" in item["input"].lower())
    frustration_count = sum(1 for item in final_data if "hard" in item["input"].lower() or "frustrated" in item["input"].lower() or "give up" in item["input"].lower() or "impossible" in item["input"].lower() or "hate" in item["input"].lower())
    answer_count = sum(1 for item in final_data if "answer" in item["input"].lower() or "solution" in item["input"].lower() or "tell me" in item["input"].lower())
    formula_count = sum(1 for item in final_data if "can't remember" in item["input"].lower() or "blank" in item["input"].lower() or "don't remember" in item["input"].lower())
    
    print(f"\nScenario breakdown:")
    print(f"  - 'I don't know' scenarios: ~{idk_count}")
    print(f"  - Frustration scenarios: ~{frustration_count}")
    print(f"  - 'Give me answer' scenarios: ~{answer_count}")
    print(f"  - 'Can't remember formula' (scaffolding): ~{formula_count}")
    print(f"  - Normal progression + other: ~{len(final_data) - idk_count - frustration_count - answer_count - formula_count}")
    
    # Show sample
    print("\n" + "="*60)
    print("SAMPLE DATA POINT")
    print("="*60)
    sample = random.choice(final_data)
    print(f"\nINSTRUCTION (first 400 chars):\n{sample['instruction'][:400]}...")
    print(f"\nINPUT:\n{sample['input']}")
    print(f"\nOUTPUT:\n{sample['output']}")

if __name__ == "__main__":
    create_training_dataset("all_steps_extracted.json", "sft_with_real_hints.json")
