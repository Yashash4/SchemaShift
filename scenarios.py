"""Scenario definitions (E1-E3 easy). Medium/hard added in Phase 11."""
from __future__ import annotations

from models import DriftEvent


SCENARIOS: dict[str, dict] = {
    "E1_onboard_new_hire": {
        "difficulty": "easy",
        "max_steps": 8,
        "token_budget": 4000,
        "task_description": (
            "A new employee named Priya Sharma (priya@company.com) starts on "
            "Monday April 27. Send her a welcome email at priya@company.com "
            "with subject containing 'welcome', then create a calendar event "
            "titled 'New Hire Orientation' on Monday April 27 at 10:00 AM for "
            "1 hour, inviting both Priya and her manager alex@company.com."
        ),
        "success_criteria": [
            "Welcome email sent to priya@company.com with 'welcome' in subject",
            "Calendar event created for Monday April 27 10am with both Priya "
            "and Alex attending",
        ],
        "seed_data": {
            "mail": {"messages": []},
            "calendar": {"events": []},
        },
        "drift_plan": [
            DriftEvent(
                tool="calendar",
                endpoint="create_event",
                kind="field_rename",
                fires_at_step=3,
                details={"from": "attendees", "to": "participants"},
            ),
        ],
        "ground_truth_final_state": {
            "mail.sent_count": 1,
            "mail.last_sent_to": "priya@company.com",
            "mail.last_subject_contains_welcome": True,
            "calendar.events_count": 1,
            "calendar.last_event_has_both_attendees": True,
        },
        "required_tools": ["mail", "calendar"],
    },

    "E2_meeting_invite_blast": {
        "difficulty": "easy",
        "max_steps": 6,
        "token_budget": 4000,
        "task_description": (
            "Send calendar invite emails to three team members for an "
            "all-hands meeting at 3:00 PM Friday. Recipients: "
            "alex@company.com, jordan@company.com, sam@company.com. "
            "Subject: 'All-Hands: Friday 3pm'. "
            "Body should mention the time and agenda."
        ),
        "success_criteria": [
            "Three emails sent to the three listed recipients",
            "All three emails have subject containing 'All-Hands'",
        ],
        "seed_data": {
            "mail": {"messages": []},
        },
        "drift_plan": [
            DriftEvent(
                tool="mail",
                endpoint="send_message",
                kind="endpoint_deprecation",
                fires_at_step=1,
                details={"replacement": "messages.send"},
            ),
        ],
        "ground_truth_final_state": {
            "mail.sent_count": 3,
            "mail.sent_to_all_three_recipients": True,
        },
        "required_tools": ["mail"],
    },

    "E3_customer_lookup": {
        "difficulty": "easy",
        "max_steps": 8,
        "token_budget": 4000,
        "task_description": (
            "A customer emailed support asking about their account: "
            "bob@customer.com. Look them up in the CRM by email, retrieve "
            "their full contact record, and update their status field to "
            "'support_in_progress'. Report back their company name in your "
            "final summary."
        ),
        "success_criteria": [
            "Customer bob@customer.com found in CRM",
            "Contact status updated to 'support_in_progress'",
            "Completion summary mentions the customer's company",
        ],
        "seed_data": {
            "crm": {
                "contacts": [
                    {"contact_id": "c_1", "customer_email": "alice@customer.com",
                     "name": "Alice Nguyen", "company": "Acme Corp", "status": "active"},
                    {"contact_id": "c_2", "customer_email": "bob@customer.com",
                     "name": "Bob Taylor", "company": "Globex Industries", "status": "active"},
                    {"contact_id": "c_3", "customer_email": "carol@customer.com",
                     "name": "Carol Davis", "company": "Initech", "status": "inactive"},
                ],
            },
        },
        "drift_plan": [
            DriftEvent(
                tool="crm",
                endpoint=None,
                kind="field_rename",
                fires_at_step=2,
                details={"from": "customer_email", "to": "email_address"},
            ),
        ],
        "ground_truth_final_state": {
            "crm.contact_c_2_status": "support_in_progress",
            "complete_summary_mentions_company": True,
        },
        "required_tools": ["crm"],
    },
}
