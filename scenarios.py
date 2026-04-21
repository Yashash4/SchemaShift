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

    "M1_customer_escalation": {
        "difficulty": "medium",
        "max_steps": 12,
        "token_budget": 6000,
        "task_description": (
            "A VIP customer at bob@customer.com has escalated: their "
            "subscription is about to lapse. Look them up in CRM, update "
            "their status to 'vip_escalation', send them a personalized "
            "retention email from support@company.com with subject "
            "'Priority Support — [Customer Name]', and schedule a 30-minute "
            "check-in call on Friday April 24 at 2:00 PM with both the "
            "customer and the account manager alex@company.com."
        ),
        "success_criteria": [
            "Customer contact retrieved with correct company",
            "CRM status updated to vip_escalation",
            "Retention email sent to bob@customer.com with Priority Support subject",
            "Calendar event created for Friday April 24 2pm with both customer and account manager",
        ],
        "seed_data": {
            "mail": {"messages": []},
            "calendar": {"events": []},
            "crm": {
                "contacts": [
                    {"contact_id": "c_1", "customer_email": "alice@customer.com",
                     "name": "Alice Nguyen", "company": "Acme Corp", "status": "active"},
                    {"contact_id": "c_2", "customer_email": "bob@customer.com",
                     "name": "Bob Taylor", "company": "Globex Industries", "status": "active"},
                ],
            },
        },
        "drift_plan": [
            DriftEvent(
                tool="crm", endpoint=None, kind="field_rename",
                fires_at_step=1,
                details={"from": "customer_email", "to": "email_address"},
            ),
            DriftEvent(
                tool="calendar", endpoint="create_event", kind="field_rename",
                fires_at_step=6,
                details={"from": "attendees", "to": "participants"},
            ),
        ],
        "ground_truth_final_state": {
            "crm.contact_c_2_status": "vip_escalation",
            "mail.sent_count": 1,
            "mail.last_sent_to": "bob@customer.com",
            "mail.last_subject_contains_priority_support": True,
            "calendar.events_count": 1,
            "calendar.last_event_has_both_attendees": True,
        },
        "required_tools": ["mail", "calendar", "crm"],
    },

    "M2_weekly_report": {
        "difficulty": "medium",
        "max_steps": 10,
        "token_budget": 5000,
        "task_description": (
            "Prepare the weekly sales report: pull the list of active "
            "contacts from CRM, send a summary email to "
            "sales-leads@company.com with subject "
            "'Weekly Active Contacts Report' listing contact names, and "
            "schedule a report review meeting next Monday April 27 at "
            "10:00 AM with the sales team leads sarah@company.com and "
            "mike@company.com."
        ),
        "success_criteria": [
            "Active contacts retrieved from CRM",
            "Summary email sent with 'Weekly' in subject",
            "Meeting scheduled for Monday April 27 10am with both sales leads",
        ],
        "seed_data": {
            "mail": {"messages": []},
            "calendar": {"events": []},
            "crm": {
                "contacts": [
                    {"contact_id": "c_1", "customer_email": "x@co.com",
                     "name": "X Person", "company": "Co", "status": "active"},
                    {"contact_id": "c_2", "customer_email": "y@co.com",
                     "name": "Y Person", "company": "Co", "status": "active"},
                    {"contact_id": "c_3", "customer_email": "z@co.com",
                     "name": "Z Person", "company": "Co", "status": "inactive"},
                ],
            },
        },
        "drift_plan": [
            DriftEvent(
                tool="mail", endpoint="send_message", kind="endpoint_deprecation",
                fires_at_step=2,
                details={"replacement": "messages.send"},
            ),
            DriftEvent(
                tool="crm", endpoint=None, kind="rate_limit_tightening",
                fires_at_step=4,
                details={"limit": 2},
            ),
        ],
        "ground_truth_final_state": {
            "mail.sent_count": 1,
            "mail.last_subject_contains_weekly": True,
            "calendar.events_count": 1,
            "calendar.last_event_has_both_sales_leads": True,
        },
        "required_tools": ["mail", "calendar", "crm"],
    },

    "M3_event_cleanup": {
        "difficulty": "medium",
        "max_steps": 12,
        "token_budget": 6000,
        "task_description": (
            "End-of-week calendar cleanup: find and cancel the "
            "'Old Planning Sync' event, find and cancel the "
            "'Cancelled Kickoff' event, and create a new 'Friday Wrap-up' "
            "event for Friday April 24 at 4:00 PM with the team lead "
            "alex@company.com attending. Send a notification email to "
            "team-all@company.com with subject "
            "'Calendar Updated — Friday Wrap-up Added' about the changes."
        ),
        "success_criteria": [
            "Old Planning Sync event cancelled or deleted",
            "Cancelled Kickoff event cancelled or deleted",
            "New Friday Wrap-up event created at 4pm with alex attending",
            "Notification email sent with 'Calendar Updated' in subject",
        ],
        "seed_data": {
            "mail": {"messages": []},
            "calendar": {"events": [
                {"event_id": "evt_1", "title": "Old Planning Sync",
                 "start": "2026-04-20T10:00:00Z", "end": "2026-04-20T11:00:00Z",
                 "attendees": ["alex@company.com"], "status": "confirmed"},
                {"event_id": "evt_2", "title": "Cancelled Kickoff",
                 "start": "2026-04-21T14:00:00Z", "end": "2026-04-21T15:00:00Z",
                 "attendees": ["alex@company.com"], "status": "confirmed"},
            ]},
        },
        "drift_plan": [
            DriftEvent(
                tool="calendar", endpoint="delete_event", kind="tool_removal",
                fires_at_step=2,
                details={"fallback": "update_event status=cancelled"},
            ),
            DriftEvent(
                tool="calendar", endpoint="create_event", kind="field_rename",
                fires_at_step=5,
                details={"from": "attendees", "to": "participants"},
            ),
        ],
        "ground_truth_final_state": {
            "calendar.evt_1_status": "cancelled",
            "calendar.evt_2_status": "cancelled",
            "calendar.events_count_new_friday_wrapup": 1,
            "mail.sent_count": 1,
            "mail.last_subject_contains_calendar_updated": True,
        },
        "required_tools": ["mail", "calendar"],
    },
}
