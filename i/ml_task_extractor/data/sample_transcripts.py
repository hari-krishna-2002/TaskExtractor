"""
Sample meeting transcripts for testing and demonstration
"""

SAMPLE_TRANSCRIPTS = {
    "project_meeting": """
    John: Good morning everyone. Let's start with the project status update.

    Sarah: The frontend development is progressing well. We should have the user interface complete by Friday. However, we need to review the design mockups with the client before proceeding.

    Mike: On the backend side, we've finished the API development. The database integration needs to be tested by tomorrow. I'll handle that.

    John: Great. Sarah, can you schedule a client review meeting for next week? We need their approval before the final implementation.

    Sarah: Sure, I'll book a meeting room and send out invites by end of day.

    Lisa: From the QA perspective, we need to prepare test cases. This should be done by Wednesday so we can start testing immediately after the client approval.

    John: Perfect. Let's also make sure the documentation is updated. Mike, can you handle that?

    Mike: Absolutely. I'll update the technical documentation by Thursday.

    John: One more thing - we have a hard deadline for the project delivery on March 15th. Everyone needs to be aware of this.
    """,

    "sales_meeting": """
    Manager: Let's review our Q4 targets and action items.

    Tom: I've been working on the Johnson account. They're interested but need a proposal by next Friday. I'll prepare that this week.

    Manager: Good. What about the Miller contract?

    Amy: The Miller contract is almost ready. I just need legal review, which should be complete by tomorrow. After that, we can send it to the client.

    Manager: Excellent. Tom, after you finish the Johnson proposal, I need you to follow up with the Peterson lead. They haven't responded to our last email.

    Tom: Will do. I'll call them by Wednesday.

    Manager: Amy, can you prepare the monthly sales report? We need it for the board meeting next week.

    Amy: Of course. I'll have it ready by Monday morning.

    Manager: Great. One urgent item - the pricing sheet needs to be updated immediately. There are some errors that need to be fixed before we send any more proposals.

    Tom: I can handle that. I'll update it today and send it to everyone for review.
    """,

    "technical_standup": """
    Lead: Daily standup time. Let's go around the team.

    Developer1: Yesterday I worked on the authentication module. Today I'll continue with the password reset functionality. Should be done by tomorrow.

    Developer2: I finished the database migration scripts. Today I'm starting on the API endpoints. The user management API needs to be complete by Friday.

    Lead: Any blockers?

    Developer1: I need the security requirements document. Can someone send that to me?

    Lead: I'll send that right after this meeting.

    Developer2: The staging environment is down. We need to fix that before we can deploy for testing.

    Lead: I'll contact DevOps about that immediately. In the meantime, continue with local development.

    Developer1: Also, we should schedule a code review session for next week. The authentication module will be ready for review by then.

    Lead: Good point. I'll book a conference room for Tuesday afternoon.

    Developer2: Don't forget we have the architecture review meeting on Thursday. We need to prepare the technical presentation.

    Lead: Right. I'll prepare the slides by Wednesday. Everyone should review the current architecture documentation before the meeting.
    """,

    "client_meeting": """
    Account Manager: Thank you for joining us today. Let's discuss the project timeline and deliverables.

    Client: We're excited about this project. When can we expect the first milestone?

    Project Manager: Based on our discussion, the initial prototype should be ready by January 20th. We'll need your feedback within 3 days of delivery.

    Client: That works for us. What about the final delivery?

    Project Manager: The complete system will be delivered by March 1st. However, we need the final requirements document from your team by next Friday.

    Client: I'll make sure our technical team provides that. Who should they contact?

    Account Manager: They can reach out to our technical lead, Sarah. I'll send her contact information after this meeting.

    Client: Perfect. We also need training for our staff. Can that be arranged?

    Project Manager: Absolutely. We'll schedule training sessions for the week after delivery. I'll coordinate with your HR team to set up the schedule.

    Client: Great. One more thing - we need a detailed project plan with all milestones and deadlines.

    Account Manager: I'll prepare that document and send it to you by tomorrow. It will include all deliverables, timelines, and responsible parties.

    Client: Excellent. We look forward to working with you.
    """,

    "emergency_meeting": """
    Director: We have a critical situation. The production server went down an hour ago.

    DevOps: I'm already working on it. The issue seems to be with the database connection. I need to restore from backup immediately.

    Director: How long will that take?

    DevOps: About 2 hours for full restoration. But I can get a temporary fix running in 30 minutes.

    Director: Do the temporary fix first. We need the system back online ASAP.

    Support Lead: I'm getting calls from customers. Should I send out a status update?

    Director: Yes, send an immediate notification to all customers explaining the situation and expected resolution time.

    DevOps: I also need to investigate the root cause after the restoration. This shouldn't have happened.

    Director: Agreed. Schedule a post-mortem meeting for tomorrow morning. We need to prevent this from happening again.

    Support Lead: I'll document all customer complaints and issues for the post-mortem.

    Director: Good. Let's reconvene in one hour to check the progress. Everyone stay focused on getting us back online.

    DevOps: Understood. I'll provide updates every 15 minutes.
    """,

    "planning_meeting": """
    Product Manager: Let's plan the next sprint. We have 10 story points to allocate.

    Developer: The user authentication feature is estimated at 5 points. I can complete that by next Wednesday.

    Designer: The UI mockups for the dashboard need to be finalized. That's about 3 points and should be done by Tuesday.

    Product Manager: Great. What about the payment integration?

    Developer: That's complex - probably 8 points. It might spill over to the next sprint.

    Product Manager: Let's break it down. Can we do the basic integration this sprint and advanced features next sprint?

    Developer: Yes, basic payment processing would be 5 points. I can finish that by Friday.

    QA Engineer: I need to prepare test cases for all these features. The authentication tests should be ready by Thursday.

    Product Manager: Perfect. Designer, after you finish the dashboard mockups, can you start on the payment flow designs?

    Designer: Sure, I'll have those ready by end of week.

    Product Manager: Excellent. Let's also schedule a demo for stakeholders next Friday. Everyone should have their features ready for presentation.

    QA Engineer: I'll make sure all features are tested before the demo.
    """
}

# Meeting contexts for testing
MEETING_CONTEXTS = {
    "project_meeting": "Weekly Project Status Meeting - Q1 Planning",
    "sales_meeting": "Q4 Sales Review and Planning",
    "technical_standup": "Daily Development Team Standup",
    "client_meeting": "Client Kickoff Meeting - Project Alpha",
    "emergency_meeting": "Emergency Response - Production Outage",
    "planning_meeting": "Sprint Planning - Development Team"
}

# Expected tasks for validation (for testing purposes)
EXPECTED_TASKS = {
    "project_meeting": [
        "Complete user interface by Friday",
        "Review design mockups with client",
        "Test database integration by tomorrow",
        "Schedule client review meeting for next week",
        "Book meeting room and send invites by end of day",
        "Prepare test cases by Wednesday",
        "Update technical documentation by Thursday",
        "Project delivery deadline March 15th"
    ],

    "sales_meeting": [
        "Prepare Johnson proposal by next Friday",
        "Complete legal review by tomorrow",
        "Follow up with Peterson lead by Wednesday",
        "Prepare monthly sales report by Monday morning",
        "Update pricing sheet today"
    ]
}


def get_sample_transcript(meeting_type: str) -> str:
    """Get a sample transcript by type."""
    return SAMPLE_TRANSCRIPTS.get(meeting_type, "")


def get_meeting_context(meeting_type: str) -> str:
    """Get meeting context by type."""
    return MEETING_CONTEXTS.get(meeting_type, "")


def get_expected_tasks(meeting_type: str) -> list:
    """Get expected tasks for validation."""
    return EXPECTED_TASKS.get(meeting_type, [])