# 🏫 SMART CLASSROOM SIMULATOR - Collections + OOP Mastery!
import random
import time
from datetime import datetime

class Person:
    """Base class for all people in the classroom"""
    def __init__(self, name):
        self.name = name
        self.id = f"P_{random.randint(1000, 9999)}"
    
    def introduce(self):
        return f"Hi, I'm {self.name}"
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.name}"

class Student(Person):
    """Student class inheriting from Person - demonstrates INHERITANCE"""
    def __init__(self, name, initial_skill=None):
        super().__init__(name)  # Call parent constructor
        self.skill_level = initial_skill or random.randint(20, 60)
        self.grades = []  # LIST: grade history
        self.learned_skills = set()  # SET: unique skills mastered
        self.attendance = 100
        self.join_date = datetime.now()
        self.total_lessons = 0
    
    def study(self, skill_topic):
        """Student-specific method"""
        # Realistic learning: diminishing returns
        current = self.skill_level
        max_gain = (100 - current) * 0.15
        actual_gain = random.uniform(0.5, max_gain)
        
        self.skill_level = min(current + actual_gain, 99)
        self.learned_skills.add(skill_topic)
        self.total_lessons += 1
        
        return f"📚 {self.name} studied {skill_topic}: {current:.1f}% → {self.skill_level:.1f}%"
    
    def take_test(self):
        """Take a test - combines skill with randomness"""
        skill_component = self.skill_level * 0.7
        random_component = random.uniform(-15, 15)  # Test day factors
        
        grade = max(0, min(100, skill_component + random_component))
        self.grades.append(grade)
        
        return grade
    
    def get_average_grade(self):
        """Calculate student's average grade"""
        if not self.grades:
            return 0
        return sum(self.grades) / len(self.grades)
    
    def introduce(self):
        """Override parent method - demonstrates POLYMORPHISM"""
        return f"👨‍🎓 Hi, I'm {self.name}, a student with {self.skill_level:.1f}% skill level"

class Teacher(Person):
    """Teacher class inheriting from Person"""
    def __init__(self, name, subject):
        super().__init__(name)
        self.subject = subject
        self.lessons_taught = 0
    
    def teach(self, students, skill_topic):
        """Teach a lesson to all students"""
        results = []
        for student in students:
            result = student.study(skill_topic)
            results.append(result)
        
        self.lessons_taught += 1
        return results
    
    def introduce(self):
        """Override parent method - demonstrates POLYMORPHISM"""
        return f"👨‍🏫 Hi, I'm Prof. {self.name}, I teach {self.subject}"

class SmartClassroom:
    """
    Advanced classroom management system combining:
    - Collections (Lists, Dicts, Sets, Tuples)
    - OOP (Classes, Inheritance, Polymorphism, Encapsulation)
    """
    
    def __init__(self, class_name="OOP Mastery Classroom"):
        self.class_name = class_name
        self.students = []  # LIST of Student objects
        self.student_registry = {}  # DICT: student_id -> Student object
        self.teachers = []  # LIST of Teacher objects
        self.completed_lessons = set()  # SET: unique lessons taught
        self.schedule = (  # TUPLE: immutable timetable
            ("09:00", "Morning Lesson"),
            ("10:30", "Practice Time"),
            ("11:00", "Quiz Time"),
            ("12:00", "Break Time")
        )
        self.day_count = 1
        
        # Add a default teacher
        self.add_teacher("Dr. Python", "Programming")
        
    def add_student(self, name, initial_skill=None):
        """Add a new student using OOP principles"""
        student = Student(name, initial_skill)
        
        # Add to collections
        self.students.append(student)  # LIST
        self.student_registry[student.id] = student  # DICT
        
        print(f"🎉 {student.introduce()}")
        print(f"   Student ID: {student.id}")
        return student
    
    def add_teacher(self, name, subject):
        """Add a new teacher"""
        teacher = Teacher(name, subject)
        self.teachers.append(teacher)
        print(f"👨‍🏫 {teacher.introduce()} joined the classroom!")
        return teacher
    
    def teach_lesson(self, skill_topic):
        """Conduct a lesson using OOP design"""
        if not self.students:
            print("📚 Empty classroom - no students to teach!")
            return
        
        print(f"\n🎓 Teaching lesson: '{skill_topic.upper()}' - Day {self.day_count}")
        print("=" * 60)
        
        # Add to completed lessons SET
        self.completed_lessons.add(skill_topic)
        
        # Get the first available teacher
        teacher = self.teachers[0] if self.teachers else None
        
        if teacher:
            # Teacher teaches all students (OOP in action!)
            results = teacher.teach(self.students, skill_topic)
            for result in results:
                print(f"  {result}")
        else:
            # Fallback: students study independently
            for student in self.students:
                result = student.study(skill_topic)
                print(f"  {result}")
        
        time.sleep(1)  # Simulate teaching time
    
    def conduct_test(self):
        """Run a test for all students"""
        if not self.students:
            print("📝 No students to test!")
            return
        
        print(f"\n📝 QUIZ TIME - Day {self.day_count}")
        print("=" * 60)
        
        test_results = []
        
        for student in self.students:
            grade = student.take_test()
            avg_grade = student.get_average_grade()
            
            test_results.append((student.name, grade, avg_grade))
            print(f"  📊 {student.name}: {grade:.1f} (avg: {avg_grade:.1f})")
        
        # Find top performer
        if test_results:
            best_student = max(test_results, key=lambda x: x[1])
            print(f"\n🏆 Top scorer: {best_student[0]} with {best_student[1]:.1f}!")
        
        self.day_count += 1
    
    def class_statistics(self):
        """Show comprehensive statistics using OOP data"""
        if not self.students:
            print("📊 No students enrolled!")
            return
        
        print(f"\n📊 CLASS STATISTICS - {self.class_name}")
        print("=" * 70)
        
        # Calculate overall statistics
        total_students = len(self.students)
        avg_skill = sum(s.skill_level for s in self.students) / total_students
        
        # Students with grades
        graded_students = [s for s in self.students if s.grades]
        if graded_students:
            all_grades = [grade for s in graded_students for grade in s.grades]
            avg_grade = sum(all_grades) / len(all_grades)
            print(f"📈 Class average skill: {avg_skill:.1f}%")
            print(f"📝 Class average grade: {avg_grade:.1f}")
        else:
            print(f"📈 Class average skill: {avg_skill:.1f}%")
            print(f"📝 No grades yet")
        
        print(f"👥 Total students: {total_students}")
        print(f"👨‍🏫 Teachers: {len(self.teachers)}")
        print(f"📚 Lessons completed: {len(self.completed_lessons)}")
        print(f"🎯 Skills taught: {', '.join(self.completed_lessons)}")
        
        # Individual student progress (POLYMORPHISM in action)
        print(f"\n👥 INDIVIDUAL PROGRESS:")
        for student in self.students:
            skills_count = len(student.learned_skills)
            avg_grade = student.get_average_grade()
            
            print(f"  • {student.name}: {student.skill_level:.1f}% skill, "
                  f"avg grade: {avg_grade:.1f}, {skills_count} skills mastered")
        
        # Teacher statistics
        print(f"\n👨‍🏫 TEACHING STAFF:")
        for teacher in self.teachers:
            print(f"  • {teacher.introduce()} - Lessons taught: {teacher.lessons_taught}")
    
    def demonstrate_polymorphism(self):
        """Demonstrate polymorphism with all people in classroom"""
        print(f"\n🎭 POLYMORPHISM DEMONSTRATION:")
        print("Same method, different behaviors:")
        print("-" * 40)
        
        # All people (students + teachers) have introduce() method
        all_people = self.students + self.teachers
        
        for person in all_people:
            print(f"  {person.introduce()}")  # Polymorphism in action!
    
    def remove_student(self, student_id):
        """Remove a student by ID"""
        if student_id in self.student_registry:
            student = self.student_registry[student_id]
            self.students.remove(student)
            del self.student_registry[student_id]
            print(f"👋 {student.name} has left the class. Good luck!")
        else:
            print(f"❌ Student with ID {student_id} not found!")

# 🎮 INTERACTIVE OOP SIMULATION
def run_oop_classroom_simulation():
    """Run the complete OOP classroom experience!"""
    classroom = SmartClassroom("OOP Foundations Bootcamp")
    
    print("🏫 WELCOME TO THE SMART OOP CLASSROOM SIMULATOR!")
    print("Experience the power of Collections + OOP working together!\n")
    
    # Add students using OOP
    students = [
        classroom.add_student("Ana Popescu", 45),
        classroom.add_student("Mihai Ionescu", 35),
        classroom.add_student("Elena Vasilescu", 50),
        classroom.add_student("Andrei Marin", 40)
    ]
    
    # Demonstrate polymorphism
    classroom.demonstrate_polymorphism()
    
    # Teaching sequence
    lessons = ["variables", "if_else_statements", "loops", "collections", "oop"]
    
    for day, lesson in enumerate(lessons, 1):
        print(f"\n🌅 DAY {day} - Teaching {lesson.replace('_', ' ').title()}")
        classroom.teach_lesson(lesson)
        classroom.conduct_test()
        classroom.class_statistics()
        
        if day == 3:  # Add a student mid-course
            classroom.add_student("Maria Georgescu", 55)
            classroom.demonstrate_polymorphism()  # Show updated polymorphism
        
        print("\n" + "⭐" * 80)
        time.sleep(1)  # Brief pause between days
    
    print("\n🎓 OOP SIMULATION COMPLETE!")
    print("You've experienced how Collections and OOP create powerful, living systems!")

# Interactive Menu System
def interactive_oop_classroom():
    """Interactive classroom management with OOP"""
    classroom = SmartClassroom("Interactive OOP Classroom")
    
    while True:
        print("\n🏫 OOP CLASSROOM MANAGEMENT MENU")
        print("1. Add Student")
        print("2. Add Teacher")
        print("3. Remove Student") 
        print("4. Teach Lesson")
        print("5. Conduct Test")
        print("6. View Statistics")
        print("7. Demonstrate Polymorphism")
        print("8. Full OOP Simulation")
        print("0. Exit")
        
        choice = input("Choose an option: ").strip()
        
        if choice == "1":
            name = input("Student name: ").strip()
            skill = input("Initial skill (press Enter for random): ").strip()
            skill_level = int(skill) if skill.isdigit() else None
            if name:
                classroom.add_student(name, skill_level)
                
        elif choice == "2":
            name = input("Teacher name: ").strip()
            subject = input("Subject: ").strip()
            if name and subject:
                classroom.add_teacher(name, subject)
                
        elif choice == "3":
            if classroom.students:
                print("Students:")
                for i, student in enumerate(classroom.students):
                    print(f"  {i}: {student.name} (ID: {student.id})")
                try:
                    idx = int(input("Student index to remove: "))
                    if 0 <= idx < len(classroom.students):
                        student_id = classroom.students[idx].id
                        classroom.remove_student(student_id)
                except (ValueError, IndexError):
                    print("Invalid index")
            else:
                print("No students in class!")
                
        elif choice == "4":
            skill = input("Skill to teach (e.g., 'oop'): ").strip()
            if skill:
                classroom.teach_lesson(skill)
                
        elif choice == "5":
            classroom.conduct_test()
            
        elif choice == "6":
            classroom.class_statistics()
            
        elif choice == "7":
            classroom.demonstrate_polymorphism()
            
        elif choice == "8":
            run_oop_classroom_simulation()
            
        elif choice == "0":
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice!")

# Run the OOP classroom experience
if __name__ == "__main__":
    print("🎓 OOP CLASSROOM SIMULATOR MODES:")
    print("1. Interactive Menu")
    print("2. Auto OOP Simulation")
    
    mode = input("Choose mode (1 or 2): ").strip()
    
    if mode == "1":
        interactive_oop_classroom()
    else:
        run_oop_classroom_simulation()