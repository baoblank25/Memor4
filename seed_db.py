"""
seed_db.py - Database Initializer for Memento Protocol
Creates an empty database ready to add users and memories.
"""

import sys
from colorama import Fore, Style, init

init(autoreset=True)

from memory import (
    init_db,
    clear_all_data,
    register_person,
    save_memory,
    get_all_people,
    get_memory_stats,
    reset_memory_status
)


def seed_database(clear_first: bool = False):
    """
    Initialize database with empty tables (no demo users or memories).
    """
    print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   MEMENTO PROTOCOL - Database Seeder{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}\n")

    # Initialize
    init_db()

    if clear_first:
        print(f"{Fore.YELLOW}Clearing existing data...{Style.RESET_ALL}\n")
        clear_all_data()

    print(f"{Fore.GREEN}Database initialized with empty tables (no user profiles or demo data).{Style.RESET_ALL}\n")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print(f"\n{Fore.GREEN}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}   DATABASE INITIALIZED SUCCESSFULLY{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*50}{Style.RESET_ALL}\n")

    # Stats
    stats = get_memory_stats()
    print(f"  📊 {stats['total_people']} people registered")
    print(f"  💭 {stats['total_memories']} memories stored")
    print()

    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   GETTING STARTED{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}\n")

    print(f"  1. Start server:  {Fore.GREEN}python server.py{Style.RESET_ALL}")
    print(f"  2. Open browser:  {Fore.GREEN}http://localhost:8000{Style.RESET_ALL}")
    print(f"  3. Add family members and start creating memories!")
    print()


def main():
    clear_first = '--clear' in sys.argv
    
    if clear_first:
        print(f"\n{Fore.YELLOW}⚠ --clear flag: Will reset database{Style.RESET_ALL}")
    
    seed_database(clear_first=clear_first)


if __name__ == "__main__":
    main()